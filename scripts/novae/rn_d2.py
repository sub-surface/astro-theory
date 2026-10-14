"""EXP-2026-Z H_ML: train and evaluate D2 (learned detector) — locally or on one Modal B200.

Protocol (pre-registered in docs/programmes/novae.md):
  train  : control light curves of the train-half tids (packed.npz `train`), windows +-3 yr, 50% with a synthetic
           eruption injected on the fly (label = >= 2 injected detections), BCE.
  test   : held-out controls (test-half tids): light-curve score = max over windows (centres every 0.25 yr);
           positives: the real catalogued eruptions in data/rn/recall.csv (target light curves, never trained on):
           max over windows centred inside the eruption's date window (every 0.1 yr).
Outputs data/rn/d2_eval.npz (control scores, positive scores) for rn_analyse.py.

  local : python scripts/novae/rn_d2.py --steps 300 --batch 64
  modal : $env:PYTHONIOENCODING="utf-8"; $env:PYTHONUTF8=1; modal run scripts/novae/rn_d2.py --batch 1024 --budget-s 420
"""
import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
ROOT = _HERE.parents[2] if len(_HERE.parents) > 2 else _HERE.parent  # inside the Modal container: /root
RN = ROOT / "data" / "rn"


def run(npz_bytes: bytes, recall_csv: str, steps: int, batch: int, lr: float = 3e-4, seed: int = 0, log=print,
        budget_s: float = 0.0):
    """Train D2 and score held-out controls and real eruptions. budget_s > 0 caps training wall time: the step
    count is fixed after timing 20 probe steps, so the one-cycle schedule still completes."""
    import pandas as pd
    import torch
    try:
        from celestrium.novae import learned as Lr
    except ImportError:  # inside the Modal container
        import learned as Lr
    torch.manual_seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    npz = dict(np.load(io.BytesIO(npz_bytes), allow_pickle=False))
    role, train = npz["role"], npz["train"]
    tr = Lr.Packed(npz, dev, keep=(role != "t") & train)
    model = Lr.D2().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    use_amp = dev.type == "cuda"
    t_start = time.time()
    n = len(tr.index)

    def train_step(step, sched=None):
        lc = torch.randint(0, n, (batch,), device=dev)
        span = (tr.tmax[lc] - tr.tmin[lc]).clamp(min=0.1)
        center = tr.tmin[lc] + torch.rand(batch, device=dev) * span
        w, mask = tr.windows(lc, center)
        m_q = Lr.quiescent(w, mask, tr.m_faint[lc])
        m_q = torch.nan_to_num(m_q, nan=20.0)
        inj = torch.rand(batch, device=dev) < 0.5
        w_inj, lab = Lr.inject(w, mask, m_q, center)
        w = torch.where(inj[:, None, None], w_inj, w)
        y = torch.where(inj, lab, torch.zeros_like(lab))
        x = Lr.features(w, mask, m_q, center)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=use_amp):
            logit = model(x, mask)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit.float(), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if sched is not None:
            sched.step()
        return loss, y

    if budget_s > 0:  # probe the step time at a low constant learning rate, then size the run to the budget
        for g in opt.param_groups:
            g["lr"] = lr / 25
        for k in range(20):
            train_step(k)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        per = (time.time() - t_start) / 20
        steps = int(max(100, min(steps, (budget_s - (time.time() - t_start)) / per)))
        log(f"probe: {per * 1e3:.1f} ms/step -> {steps} steps within {budget_s:.0f}s")
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    for step in range(steps):
        loss, y = train_step(step, sched)
        if step % max(1, steps // 20) == 0 or step == steps - 1:
            log(f"step {step} loss {loss.item():.4f} pos {y.mean().item():.2f} {time.time() - t_start:.0f}s")
    model.eval()

    @torch.no_grad()
    def score_windows(P, lc_idx, centers):
        out = []
        chunk = 2048 if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 40e9 else 128
        for s in range(0, len(lc_idx), chunk):
            lc = torch.tensor(lc_idx[s:s + chunk], device=dev)
            c = torch.tensor(centers[s:s + chunk], device=dev, dtype=torch.float32)
            w, mask = P.windows(lc, c)
            m_q = torch.nan_to_num(Lr.quiescent(w, mask, P.m_faint[lc]), nan=20.0)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=use_amp):
                out.append(model(Lr.features(w, mask, m_q, c), mask).float().cpu().numpy())
        return np.concatenate(out) if out else np.array([])

    # held-out controls: max over windows
    te = Lr.Packed(npz, dev, keep=(role != "t") & ~train)
    lc_idx, centers = [], []
    for i in range(len(te.index)):
        for c in np.arange(float(te.tmin[i]) + 0.5, float(te.tmax[i]) - 0.5, 0.25):
            lc_idx.append(i)
            centers.append(c)
    sc = score_windows(te, np.array(lc_idx), np.array(centers))
    ctrl = np.full(len(te.index), -np.inf)
    np.maximum.at(ctrl, np.array(lc_idx), sc)
    # positives: real catalogued eruptions on target light curves
    R = pd.read_csv(io.StringIO(recall_csv))
    tg = Lr.Packed(npz, dev, keep=(role == "t"))
    pos_of = {npz["tid"][j]: k for k, j in enumerate(tg.index)}
    lc_idx, centers, rid = [], [], []
    for r_i, r in enumerate(R.itertuples()):
        if r.tid not in pos_of:
            continue
        for c in np.arange(r.lo, r.hi + 1e-6, 0.1):
            lc_idx.append(pos_of[r.tid])
            centers.append(c)
            rid.append(r_i)
    sp = score_windows(tg, np.array(lc_idx), np.array(centers))
    pos = np.full(len(R), np.nan)
    for k, r_i in enumerate(rid):
        pos[r_i] = sp[k] if np.isnan(pos[r_i]) else max(pos[r_i], sp[k])
    ctrl_tid = npz["tid"][te.index]
    ctrl_role = npz["role"][te.index]
    buf = io.BytesIO()
    np.savez(buf, ctrl=ctrl, ctrl_tid=ctrl_tid, ctrl_role=ctrl_role, pos=pos)
    return buf.getvalue(), {"steps": steps, "batch": batch, "budget_s": budget_s, "seconds": time.time() - t_start,
                            "n_train_lcs": int(n), "n_test_controls": int(len(te.index)), "device": str(dev)}


try:
    import modal
    app = modal.App("celestrium-rn-d2")
    image = (modal.Image.debian_slim(python_version="3.12").pip_install("numpy", "pandas")
             .pip_install("torch==2.7.1", index_url="https://download.pytorch.org/whl/cu128")  # sm_100 (B200) kernels
             .add_local_file(str(ROOT / "celestrium" / "novae" / "learned.py"), "/root/learned.py"))

    @app.function(gpu="B200", image=image, timeout=1200)
    def remote(npz_bytes: bytes, recall_csv: str, steps: int, batch: int, budget_s: float):
        sys.path.insert(0, "/root")
        return run(npz_bytes, recall_csv, steps, batch, budget_s=budget_s)

    @app.local_entrypoint()
    def modal_main(steps: int = 20000, batch: int = 1024, budget_s: float = 0.0):
        res, info = remote.remote((RN / "packed.npz").read_bytes(), (RN / "recall.csv").read_text(), steps, batch,
                                  budget_s)
        (RN / "d2_eval.npz").write_bytes(res)
        (RN / "d2_info.json").write_text(json.dumps(info, indent=1))
        print(info)
except ImportError:
    pass


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--budget_s", type=float, default=0.0)
    a = ap.parse_args()
    res, info = run((RN / "packed.npz").read_bytes(), (RN / "recall.csv").read_text(), a.steps, a.batch,
                    budget_s=a.budget_s)
    (RN / "d2_eval.npz").write_bytes(res)
    (RN / "d2_info.json").write_text(json.dumps(info, indent=1))
    print(info)
