#!/usr/bin/env python3
"""
Fetch NVSS (1.4 GHz radio) sources via VizieR TAP — a third, independent catalogue.
Homogenisation per the 2025 kinematic paper: flux cut S1.4 > 15 mJy. NVSS covers
Dec > -40 deg; that footprint + |b|>30 are handled at the map/mask stage.
~420k sources -> one async query, written to data/nvss/nvss_S15.csv (resumable: skip if present).
"""
import os
import pyvo

OUT = "data/nvss/"
os.makedirs(OUT, exist_ok=True)
PATH = os.path.join(OUT, "nvss_S15.csv")
SVC = pyvo.dal.TAPService("https://tapvizier.cds.unistra.fr/TAPVizieR/tap")
Q = 'SELECT RAJ2000, DEJ2000, "S1.4" AS s14 FROM "VIII/65/nvss" WHERE "S1.4" > 15'

if os.path.exists(PATH) and os.path.getsize(PATH) > 100:
    print(f"[skip] {PATH} exists")
else:
    job = SVC.submit_job(Q)
    job.run()
    job.wait(phases=["COMPLETED", "ERROR", "ABORTED"], timeout=1800)
    print("phase:", job.phase)
    tab = job.fetch_result().to_table()
    tab.write(PATH, format="ascii.csv", overwrite=True)
    try:
        job.delete()
    except Exception:
        pass
    print(f"wrote {PATH}: {len(tab):,} rows")
