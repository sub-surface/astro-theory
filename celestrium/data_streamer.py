"""
=============================================================================
AstroDataStreamer: Memory-Efficient Streaming Engine for Survey Astronomy
=============================================================================
Streams multi-band photometric, astrometric, and time-domain astronomical
phenomena with full measurement error bars, presence masks, and class stratification.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Iterator, Tuple

import numpy as np
import torch
from torch.utils.data import IterableDataset

from .foundation_astrojev import (
    PHENOMENA_CLASSES,
    PHENOMENA_CLASS_TO_IDX,
    NUM_PHENOMENA_CLASSES,
    NUM_OBSERVABLES,
    NUM_UNCERTAINTIES,
    NUM_MASKS,
)


@dataclass
class PhenomenaRecord:
    """Individual astronomical phenomenon with observables, error bars, and mask."""
    source_id: str
    label: int
    class_name: str
    mu: np.ndarray      # (10,) [phot_g, bp_rp, g_bp, w1, w1_w2, pm, pm_err, l, b, snr]
    sigma: np.ndarray   # (6,)  [sig_g, sig_bprp, sig_w1, sig_w1w2, sig_pm, ruwe]
    mask: np.ndarray    # (6,)  [has_opt, has_bp, has_rp, has_w1, has_w2, has_pm]


class SyntheticPhenomenaGenerator:
    """Physically accurate simulation generator for 12 astronomical phenomena classes."""
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.counter = 0

    def generate_sample(self, class_idx: Optional[int] = None) -> PhenomenaRecord:
        """Generate a single physically consistent astronomical phenomenon record."""
        self.counter += 1
        if class_idx is None:
            # Stratified class distribution
            class_idx = int(self.rng.integers(0, NUM_PHENOMENA_CLASSES))

        src_id = f"SRC_{class_idx:02d}_{self.counter:07d}"
        c_name = PHENOMENA_CLASSES[class_idx]

        # Coordinates on sky
        ra = float(self.rng.uniform(0.0, 360.0))
        dec = float(np.degrees(np.arcsin(self.rng.uniform(-1.0, 1.0))))
        gal_l = (ra + 60.0) % 360.0
        gal_b = np.clip(dec + 15.0, -90.0, 90.0)

        # Base presence mask (all active by default)
        mask = np.ones(NUM_MASKS, dtype=np.float32)

        # ------------------------------------------------------------------- #
        # Physical Astrophysics Archetypes
        # ------------------------------------------------------------------- #
        if class_idx == 0:
            # High_z_Quasar_AGN: Faint optical, zero PM, red W1-W2 > 0.8
            phot_g = float(self.rng.uniform(18.5, 21.5))
            bp_rp = float(self.rng.normal(0.45, 0.20))
            g_bp = float(self.rng.normal(-0.25, 0.15))
            w1 = phot_g - float(self.rng.normal(4.2, 0.6))
            w1_w2 = float(self.rng.normal(1.10, 0.18))
            pm = float(self.rng.exponential(0.2))
            pm_err = float(self.rng.uniform(0.1, 0.4))
            snr = float(self.rng.uniform(15.0, 60.0))
            ruwe = float(self.rng.normal(1.05, 0.10))

        elif class_idx == 1:
            # Low_z_Seyfert_AGN: Intermediate colors, host galaxy light
            phot_g = float(self.rng.uniform(16.5, 19.5))
            bp_rp = float(self.rng.normal(0.85, 0.25))
            g_bp = float(self.rng.normal(-0.40, 0.20))
            w1 = phot_g - float(self.rng.normal(3.0, 0.5))
            w1_w2 = float(self.rng.normal(0.85, 0.15))
            pm = float(self.rng.exponential(0.3))
            pm_err = float(self.rng.uniform(0.08, 0.3))
            snr = float(self.rng.uniform(30.0, 100.0))
            ruwe = float(self.rng.normal(1.15, 0.15))

        elif class_idx == 2:
            # Luminous_Red_Galaxy (LRG): Red optical, 4000A break, zero PM
            phot_g = float(self.rng.uniform(17.5, 20.5))
            bp_rp = float(self.rng.normal(1.85, 0.25))
            g_bp = float(self.rng.normal(-0.85, 0.20))
            w1 = phot_g - float(self.rng.normal(2.8, 0.4))
            w1_w2 = float(self.rng.normal(0.40, 0.15))
            pm = float(self.rng.exponential(0.2))
            pm_err = float(self.rng.uniform(0.15, 0.5))
            snr = float(self.rng.uniform(20.0, 80.0))
            ruwe = float(self.rng.normal(1.20, 0.20))

        elif class_idx == 3:
            # Emission_Line_Galaxy (ELG): Starburst, flat SED
            phot_g = float(self.rng.uniform(19.0, 22.0))
            bp_rp = float(self.rng.normal(0.55, 0.30))
            g_bp = float(self.rng.normal(-0.30, 0.20))
            w1 = phot_g - float(self.rng.normal(2.0, 0.5))
            w1_w2 = float(self.rng.normal(0.30, 0.15))
            pm = float(self.rng.exponential(0.25))
            pm_err = float(self.rng.uniform(0.2, 0.6))
            snr = float(self.rng.uniform(10.0, 45.0))
            ruwe = float(self.rng.normal(1.10, 0.15))

        elif class_idx == 4:
            # Blazar_Relativistic_Jet: Flat spectrum, high variability
            phot_g = float(self.rng.uniform(15.5, 19.5))
            bp_rp = float(self.rng.normal(0.70, 0.35))
            g_bp = float(self.rng.normal(-0.35, 0.25))
            w1 = phot_g - float(self.rng.normal(3.8, 0.8))
            w1_w2 = float(self.rng.normal(1.25, 0.25))
            pm = float(self.rng.exponential(0.2))
            pm_err = float(self.rng.uniform(0.1, 0.3))
            snr = float(self.rng.uniform(25.0, 110.0))
            ruwe = float(self.rng.normal(1.30, 0.25))

        elif class_idx == 5:
            # Main_Sequence_Dwarf: Foreground stars, significant PM, dwarf locus
            phot_g = float(self.rng.uniform(14.0, 20.0))
            bp_rp = float(self.rng.normal(1.20, 0.40))
            g_bp = float(self.rng.normal(-0.55, 0.20))
            w1 = phot_g - float(self.rng.normal(1.2, 0.4))
            w1_w2 = float(self.rng.normal(0.10, 0.10))
            pm = float(self.rng.exponential(12.0) + 2.0)
            pm_err = float(self.rng.uniform(0.05, 0.4))
            snr = float(self.rng.uniform(40.0, 150.0))
            ruwe = float(self.rng.normal(1.00, 0.08))

        elif class_idx == 6:
            # Red_Giant_Branch: Bright, red, low PM relative to brightness
            phot_g = float(self.rng.uniform(11.0, 16.0))
            bp_rp = float(self.rng.normal(1.45, 0.30))
            g_bp = float(self.rng.normal(-0.65, 0.20))
            w1 = phot_g - float(self.rng.normal(1.8, 0.5))
            w1_w2 = float(self.rng.normal(0.15, 0.12))
            pm = float(self.rng.exponential(4.0) + 0.5)
            pm_err = float(self.rng.uniform(0.02, 0.15))
            snr = float(self.rng.uniform(80.0, 200.0))
            ruwe = float(self.rng.normal(1.02, 0.08))

        elif class_idx == 7:
            # White_Dwarf: Very blue, high reduced proper motion H_G > 15
            phot_g = float(self.rng.uniform(16.0, 20.5))
            bp_rp = float(self.rng.normal(0.05, 0.25))
            g_bp = float(self.rng.normal(0.10, 0.15))
            w1 = phot_g - float(self.rng.normal(0.2, 0.3))
            w1_w2 = float(self.rng.normal(-0.05, 0.10))
            pm = float(self.rng.exponential(25.0) + 10.0)  # High PM!
            pm_err = float(self.rng.uniform(0.1, 0.5))
            snr = float(self.rng.uniform(25.0, 90.0))
            ruwe = float(self.rng.normal(1.01, 0.07))

        elif class_idx == 8:
            # Subdwarf_Halo_Star: Fast halo kinematics, metal-poor color offset
            phot_g = float(self.rng.uniform(15.0, 19.5))
            bp_rp = float(self.rng.normal(0.75, 0.20))
            g_bp = float(self.rng.normal(-0.35, 0.15))
            w1 = phot_g - float(self.rng.normal(0.9, 0.3))
            w1_w2 = float(self.rng.normal(0.05, 0.08))
            pm = float(self.rng.exponential(45.0) + 20.0)  # Extreme halo velocity
            pm_err = float(self.rng.uniform(0.1, 0.4))
            snr = float(self.rng.uniform(30.0, 100.0))
            ruwe = float(self.rng.normal(1.02, 0.08))

        elif class_idx == 9:
            # Brown_Dwarf_LTY: Ultracool, optical dropout (very faint G), red W1-W2 > 1.8
            phot_g = float(self.rng.uniform(20.5, 23.0))
            bp_rp = float(self.rng.normal(2.50, 0.50))
            g_bp = float(self.rng.normal(-1.20, 0.40))
            w1 = phot_g - float(self.rng.normal(6.5, 0.8))
            w1_w2 = float(self.rng.normal(2.00, 0.25))
            pm = float(self.rng.exponential(30.0) + 15.0)
            pm_err = float(self.rng.uniform(0.5, 1.5))
            snr = float(self.rng.uniform(5.0, 20.0))
            ruwe = float(self.rng.normal(1.10, 0.15))
            # 50% chance of optical band dropout (missing BP/RP)
            if self.rng.uniform(0, 1) < 0.5:
                mask[1] = 0.0  # missing BP
                mask[2] = 0.0  # missing RP

        elif class_idx == 10:
            # Explosive_Transient: Rapid rise, blue optical, high SNR, zero PM
            phot_g = float(self.rng.uniform(16.5, 20.0))
            bp_rp = float(self.rng.normal(0.10, 0.30))
            g_bp = float(self.rng.normal(0.15, 0.20))
            w1 = phot_g - float(self.rng.normal(1.0, 0.8))
            w1_w2 = float(self.rng.normal(0.50, 0.30))
            pm = 0.0
            pm_err = 0.15
            snr = float(self.rng.uniform(35.0, 140.0))
            ruwe = float(self.rng.normal(1.08, 0.12))

        else:  # class_idx == 11
            # Variable_Star_Flarer: M-dwarf flares, cataclysmic variables
            phot_g = float(self.rng.uniform(15.0, 19.5))
            bp_rp = float(self.rng.normal(1.30, 0.40))
            g_bp = float(self.rng.normal(-0.60, 0.25))
            w1 = phot_g - float(self.rng.normal(1.4, 0.5))
            w1_w2 = float(self.rng.normal(0.20, 0.15))
            pm = float(self.rng.exponential(10.0) + 1.0)
            pm_err = float(self.rng.uniform(0.1, 0.5))
            snr = float(self.rng.uniform(20.0, 90.0))
            ruwe = float(self.rng.normal(1.45, 0.30))  # High RUWE due to variability

        # ------------------------------------------------------------------- #
        # Physical Heteroscedastic Error Bars
        # ------------------------------------------------------------------- #
        sig_g = float(np.clip(1.0857 / max(snr, 1.0), 0.005, 0.50))
        sig_bprp = float(np.clip(sig_g * 1.414, 0.01, 0.60))
        sig_w1 = float(np.clip(self.rng.uniform(0.02, 0.25), 0.01, 0.40))
        sig_w1w2 = float(np.clip(sig_w1 * 1.3, 0.02, 0.50))
        sig_pm = pm_err

        mu = np.array([
            phot_g, bp_rp, g_bp, w1, w1_w2,
            pm, pm_err, gal_l / 360.0, (gal_b + 90.0) / 180.0, snr
        ], dtype=np.float32)

        sigma = np.array([
            sig_g, sig_bprp, sig_w1, sig_w1w2, sig_pm, ruwe
        ], dtype=np.float32)

        return PhenomenaRecord(
            source_id=src_id,
            label=class_idx,
            class_name=c_name,
            mu=mu,
            sigma=sigma,
            mask=mask,
        )


class AstroStreamingDataset(IterableDataset):
    """PyTorch IterableDataset streaming stratified astronomical phenomena with heteroscedastic noise."""
    def __init__(
        self,
        total_samples: int = 100_000,
        batch_size: int = 256,
        seed: int = 42,
    ):
        super().__init__()
        self.total_samples = total_samples
        self.batch_size = batch_size
        self.seed = seed

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        # Distinct seed per worker process
        worker_info = torch.utils.data.get_worker_info()
        worker_seed = self.seed if worker_info is None else self.seed + worker_info.id
        gen = SyntheticPhenomenaGenerator(seed=worker_seed)

        samples_per_worker = self.total_samples
        if worker_info is not None:
            samples_per_worker = int(math.ceil(self.total_samples / worker_info.num_workers))

        n_yielded = 0
        mu_buf, sigma_buf, mask_buf, label_buf = [], [], [], []

        for _ in range(samples_per_worker):
            rec = gen.generate_sample()
            mu_buf.append(rec.mu)
            sigma_buf.append(rec.sigma)
            mask_buf.append(rec.mask)
            label_buf.append(rec.label)

            if len(mu_buf) >= self.batch_size:
                yield (
                    torch.from_numpy(np.array(mu_buf, dtype=np.float32)),
                    torch.from_numpy(np.array(sigma_buf, dtype=np.float32)),
                    torch.from_numpy(np.array(mask_buf, dtype=np.float32)),
                    torch.from_numpy(np.array(label_buf, dtype=np.int64)),
                )
                mu_buf, sigma_buf, mask_buf, label_buf = [], [], [], []
                n_yielded += self.batch_size

        if mu_buf:
            yield (
                torch.from_numpy(np.array(mu_buf, dtype=np.float32)),
                torch.from_numpy(np.array(sigma_buf, dtype=np.float32)),
                torch.from_numpy(np.array(mask_buf, dtype=np.float32)),
                torch.from_numpy(np.array(label_buf, dtype=np.int64)),
            )
