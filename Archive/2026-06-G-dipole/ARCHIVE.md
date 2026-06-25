# Archived: G — Cosmic number-count dipole / isotropy audit

**Shelved 2026-06-25.** Snapshot of the dipole line of work (project **G** in
[`../../directions.md`](../../directions.md)), parked to open the Euclid DR1-prep line.
Nothing here is abandoned — it's the methodological core we resume against DR1.

## State at shelving

Independent cross-catalogue dipole audit, milestones M1–M5 complete:

- **M1** — reproduced the published Quaia dipole (independent).
- **M2** — principled selection correction; Quaia vs. literature.
- **M3** — re-measured Ellis–Baldwin (x, α) from Quaia photometry.
- **M4** — CatWISE through the same pipeline → cross-catalogue verdict.
- **M5** — clustering-aware significance + NVSS (3rd catalogue), multipoles, redshift tomography.

Pipeline (`g_*.py`), literature dossiers + per-milestone results (`research/`),
figures (`figures/`), and local catalogue data (`data/` — git-ignored) are all here.

## Why parked

Euclid Q2 (24 Jun 2026) is the Galactic Bulge Survey — wrong geometry/population for the
dipole. The relevant Euclid dataset is **DR1 (~1900 deg² wide, 21 Oct 2026)**, the first
deep optical/NIR sample with a selection function independent of WISE/Gaia. The leverage
move is to port this pipeline to Euclid's selection function *before* DR1 and run same-day.

## To resume

`git mv` the relevant `g_*.py` back to root (paths are relative to repo root) and restore
`data/` subdirs. See [`research/results-m5.md`](./research/results-m5.md) for the live
state of the analysis.
