"""D2: a learned eruption detector for plate light curves (EXP-2026-Z, hypothesis H_ML).

Everything runs on one device in torch: windows of +-3 yr are cut from packed real light curves, synthetic
universal-decline eruptions are injected on the fly (same template and plate-detection model as D1's
injection-recovery), and a small transformer scores P(eruption in window). Only control light curves are used
for training; see scripts/novae/rn_d2.py for the protocol.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

HALF_YR = 3.0
L_MAX = 384
N_FREQ = 16
MISS = 0.02


class Packed:
    """Packed light curves on a device. Rows: t, val, det, err, lim, sat (val = mag if det else limit)."""

    def __init__(self, npz, device, keep=None):
        rows, off = npz["rows"], npz["off"]
        self.meta = {k: npz[k] for k in ("tid", "role", "m_faint", "train")}
        sel = np.arange(len(off) - 1) if keep is None else np.where(keep)[0]
        parts, offs, mf = [], [0], []
        for i in sel:
            r = rows[off[i]:off[i + 1]]
            r = r[np.argsort(r[:, 0])]
            parts.append(r)
            offs.append(offs[-1] + len(r))
            mf.append(self.meta["m_faint"][i])
        self.index = sel
        self.rows = torch.tensor(np.concatenate(parts), device=device)
        self.off = torch.tensor(offs, device=device)
        self.m_faint = torch.tensor(np.array(mf, np.float32), device=device)
        n = len(sel)
        lc_id = torch.repeat_interleave(torch.arange(n, device=device), self.off[1:] - self.off[:-1])
        self.key = lc_id.double() * 1000.0 + (self.rows[:, 0].double() - 1800.0)  # globally sorted
        self.tmin = torch.stack([self.rows[self.off[i], 0] for i in range(n)])
        self.tmax = torch.stack([self.rows[self.off[i + 1] - 1, 0] for i in range(n)])
        self.device = device

    def windows(self, lc, center, gen=None):
        """Gather up to L_MAX rows within +-HALF_YR of `center` for light curves `lc` -> (rows [B,L,6], mask [B,L])."""
        base = lc.double() * 1000.0 - 1800.0
        lo = torch.searchsorted(self.key, base + (center.double() - HALF_YR))
        hi = torch.searchsorted(self.key, base + (center.double() + HALF_YR))
        cnt = hi - lo
        ar = torch.arange(L_MAX, device=self.device)[None, :]
        u = torch.rand(len(lc), L_MAX, device=self.device, generator=gen)
        sub = torch.sort((u * cnt[:, None].clamp(min=1)).long(), dim=1).values  # random subset when crowded
        idx = torch.where(cnt[:, None] > L_MAX, lo[:, None] + sub, lo[:, None] + ar)
        mask = (ar < cnt[:, None])
        idx = torch.where(mask, idx, lo[:, None]).clamp(max=len(self.rows) - 1)
        return self.rows[idx], mask


def template_flux(dt_d, t2_d):
    tau = t2_d / (10 ** (0.8 / 1.75) - 1)
    post = (1 + dt_d.clamp(min=0) / tau) ** -1.75
    rise = 10 ** (-0.4 * 6 * (-dt_d / 3))
    return torch.where(dt_d >= 0, post, torch.where(dt_d > -3, rise, torch.zeros_like(dt_d)))


def quiescent(w, mask, m_faint):
    """Window quiescent level: median over rows of (mag if detected else max(limit, m_faint))."""
    v = torch.where(w[..., 2] > 0.5, w[..., 1], torch.maximum(w[..., 4], m_faint[:, None]))
    v = torch.where(mask, v, torch.full_like(v, float("nan")))
    return torch.nanmedian(v, dim=1).values


def inject(w, mask, m_q, center, gen=None):
    """Inject one eruption per window; returns (rows, label). Label = 1 iff >= 2 injected detections."""
    B = w.shape[0]
    dev = w.device
    t0 = center + (torch.rand(B, device=dev, generator=gen) * 2 - 1)
    t2 = torch.exp(torch.rand(B, device=dev, generator=gen) * (math.log(128) - math.log(2)) + math.log(2))
    lim = w[..., 4]
    medlim = torch.nanmedian(torch.where(mask, lim, torch.full_like(lim, float("nan"))), dim=1).values
    peak = medlim - (torch.rand(B, device=dev, generator=gen) * 6.5 - 0.5)
    dt = (w[..., 0] - t0[:, None]) * 365.25
    f = template_flux(dt, t2[:, None])
    m_inj = -2.5 * torch.log10(10 ** (-0.4 * m_q[:, None]) + 10 ** (-0.4 * peak[:, None]) * f)
    p = (1 - MISS) * 0.5 * (1 + torch.erf((lim - m_inj) / (0.3 * math.sqrt(2))))
    cur = torch.where(w[..., 2] > 0.5, w[..., 1], torch.full_like(m_inj, 99.0))
    hit = mask & (f > 1e-3) & (torch.rand(m_inj.shape, device=dev, generator=gen) < p) & (m_inj < cur)
    out = w.clone()
    out[..., 1] = torch.where(hit, m_inj + 0.2 * torch.randn(m_inj.shape, device=dev, generator=gen), w[..., 1])
    out[..., 2] = torch.where(hit, torch.ones_like(m_inj), w[..., 2])
    out[..., 3] = torch.where(hit, torch.full_like(m_inj, 0.2), w[..., 3])
    # distinct injected nights: hits separated by > 12 h (same rule as plates.n_nights)
    day = torch.where(hit, w[..., 0].double() * 365.25, torch.full(m_inj.shape, -1e9, dtype=torch.float64, device=dev))
    srt = torch.sort(day, dim=1).values
    distinct = ((srt[:, 1:] - srt[:, :-1]) > 0.5) & (srt[:, :-1] > -1e8)
    nights = distinct.sum(1) + hit.any(1).long()
    return out, (nights >= 2).float()


def features(w, mask, m_q, center):
    dt = (w[..., 0] - center[:, None]) / HALF_YR
    x = torch.stack([dt, w[..., 2], (w[..., 1] - m_q[:, None]) / 3, w[..., 3], (w[..., 4] - m_q[:, None]) / 3,
                     w[..., 5]], -1)
    x = torch.where(mask[..., None], x, torch.zeros_like(x))
    return torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)


class D2(nn.Module):
    def __init__(self, d=128, layers=4, heads=4):
        super().__init__()
        # Fourier features of time from 1/(6 yr) to 1/(1 day), so day-scale declines are representable
        freqs = torch.logspace(math.log10(2 * math.pi / 2.0), math.log10(2 * math.pi * 365.25 * 3.0), N_FREQ)
        self.register_buffer("freqs", freqs)
        self.inp = nn.Linear(6 + 2 * N_FREQ, d)
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, x, mask):
        ph = x[..., :1] * self.freqs
        h = self.inp(torch.cat([x, torch.sin(ph), torch.cos(ph)], -1))
        h = self.enc(h, src_key_padding_mask=~mask)
        m = mask[..., None].float()
        mean = (h * m).sum(1) / m.sum(1).clamp(min=1)
        mx = torch.where(mask[..., None], h, torch.full_like(h, -1e4)).max(1).values
        return self.head(torch.cat([mean, mx], -1)).squeeze(-1)
