"""Run UVOT aperture + PSF-wing photometry on all downloaded T CrB sky images (EXP-2026-Y)."""
import glob
import warnings
from pathlib import Path

import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from celestrium.novae.tcrb import uvot_aperture_photometry

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    rows = []
    for f in sorted(glob.glob(str(ROOT / "data/tcrb/uvot/*_sk.img.gz"))):
        try:
            rows += uvot_aperture_photometry(f)
        except Exception as exc:  # corrupt / partial download
            print("skip", f, exc)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "data/tcrb/uvot_photometry.csv", index=False)
    print(len(df), "snapshots;", df.groupby("filter").size().to_dict(), "saturated:", int(df.saturated.sum()))
