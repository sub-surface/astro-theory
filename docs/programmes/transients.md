# P3 · Calibrated triage engines, validated on real data

**Status**: rebuild · **Code**: [`celestrium/engines/`](../../celestrium/engines/) (evidential ML), [`celestrium/transients/`](../../celestrium/transients/) (streams, multi-messenger, FRB, GW tiling, ToO), [`celestrium/explore/`](../../celestrium/explore/) (parked, synthetic-only)
**Scripts**: [`scripts/engines/`](../../scripts/engines/), [`scripts/transients/`](../../scripts/transients/), [`scripts/explore/`](../../scripts/explore/)
**Ledger**: EXP-2026-A, D, F, H, M, N, Q, R, S, T, U, V, X, 04

## Question

Can evidential (Dirichlet) classifiers with conformal risk control make follow-up decisions that are both
useful and honestly calibrated **on the real sky**? The machinery exists (AstroJev family, RLCD calibration,
CRC gating, ToO serialisers). What does not exist yet is a single result on real, independently labelled data.

## What stands after the 2026-10-14 audit

- **EXP-G** (sound): ten live archive pipelines verified.
- Reusable machinery: Dirichlet heads, debiased calibration error, CRC routines, ToO serialisers (schema-valid), the CHIME/FRB loader.
- Everything else is synthetic-only, leaky, or withdrawn; see the audit notes in [`../ledger.md`](../ledger.md).

## Rerun queue (in priority order)

1. **Real labelled benchmark** (replaces F/H/M): Gaia DR3 + SDSS/DESI spectroscopic classes, no class-constant feature construction, no sky coordinates as features, survey-level hold-out, per-class metrics, AUROC for OOD.
2. **FRB with independent labels** (X): localised CHIME/ASKAP bursts with host redshifts as truth.
3. **Alert-stream replay** (A, R): archived ZTF/Fink packets with known classifications; field-level FDR per CLAUDE.md §3.J.1.
4. **Scheduling without oracle** (N): screening outcomes sampled from the classifier posterior; report the gain with CIs.
5. Parked until there is real data: exoplanets (Q), cosmic dawn (S), space weather (T), GW tiling (U), flow cross-calibration (V).

## Engineering debt surfaced by the audit

- `scripts/engines/modal_train_astrojev.py` silently falls back to a synthetic corpus when the catalogue read fails: make it raise.
- Missing checkpoints silently give untrained weights in several scripts (CLAUDE.md §3.O).
- Four overlapping AstroJev modules duplicate the Dirichlet KL and encoder blocks; consolidate on `engines/astrojev.py`.
- `celestrium/dashboard.py` hard-codes experiment claims (including the uncomputed "4.9σ").
- `tests/test_real_data_streaming.py` uses random arrays, not real catalogues.
