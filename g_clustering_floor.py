#!/usr/bin/env python3
"""
Clustering-aware significance (data-driven), addressing the #1 caveat that our Poisson nulls
give only shot noise.

Idea: the kinematic dipole is PURE l=1. The l>=2 multipole power therefore contains NO
kinematic signal -- it is an in-data measurement of the non-kinematic per-mode fluctuation
(clustering + residual systematics + shot). Since clustering C_l is ~flat at very low l, the
l=2 amplitude is a fair proxy for the clustering contribution to l=1. So:

  A_2_excess = sqrt(max(0, A_2^2 - <A_2,shot>^2))     # non-shot, non-kinematic per-mode power
  non-kin floor F = sqrt(<A_1,shot>^2 + A_2_excess^2)  # shot + clustering(+systematic) on l=1
  dipole significance bracket:
     optimistic (shot only):     (A_1 - <A_1,shot>) / sigma(A_1,shot)
     conservative (vs F):        A_1 / F

The truth lies between. CAVEAT: for CatWISE the residual l=2 is partly systematic (ecliptic),
so the conservative bound treats systematics as noise -> it is a LOWER bound on significance.
Reads data/multipoles_result.json (LMAX=2 run).
"""
import json
import numpy as np

M = json.load(open("data/multipoles_result.json"))
print(f"{'sample':>30} {'A_1':>7} {'A_2':>7} {'shot1':>7} {'shot2':>7} "
      f"{'non-kin F':>9} {'A1/F':>5} {'shot-sig':>8}")
for m in M:
    A = m["A"]; mu = m["null_mean"]; sd = m["null_std"]
    a1, a2 = A[1], A[2]
    mu1, mu2 = mu[1], mu[2]
    a2_exc = np.sqrt(max(0.0, a2 ** 2 - mu2 ** 2))
    F = np.sqrt(mu1 ** 2 + a2_exc ** 2)
    a1_over_F = a1 / F
    shot_sig = (a1 - mu1) / sd[1]
    print(f"{m['name']:>30} {a1:>7.4f} {a2:>7.4f} {mu1:>7.4f} {mu2:>7.4f} "
          f"{F:>9.4f} {a1_over_F:>5.2f} {shot_sig:>7.1f}")
print("\nReading: 'shot-sig' assumes only shot noise (optimistic, upper bound on significance).")
print("'A1/F' compares the dipole to a data-driven non-kinematic floor incl. clustering+residual")
print("systematics (conservative; for CatWISE the floor is systematic-inflated -> lower bound).")
print("A dipole that is genuinely kinematic+anomalous should exceed BOTH; a dipole that merely")
print("rides the general anisotropy/systematic level will have A1/F ~ 1.")
