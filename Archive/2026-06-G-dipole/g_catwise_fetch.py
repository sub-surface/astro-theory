#!/usr/bin/env python3
"""
Fetch the Secrest+2021 CatWISE2020 quasar sample via IRSA TAP, GENTLY.

Selection (Secrest+2021): AGN colour cut W1-W2 >= 0.8, magnitude 9 < W1 < 16.4 (Vega).
Galactic-plane mask |b|>30 and the ecliptic-latitude correction are applied later, at the
map stage. We pull ra, dec, w1mpro, w2mpro, w1cov, elat (ecliptic latitude, server-side).

Hardware-safe: ONE 30-deg RA stripe per query, written to its own CSV checkpoint in
data/catwise/. Resumable (skips stripes already on disk). Short cooldown between stripes.
"""
import os, time, sys
import pyvo

OUT = "data/catwise/"
os.makedirs(OUT, exist_ok=True)
SVC = pyvo.dal.TAPService("https://irsa.ipac.caltech.edu/TAP")
STRIPE = 30          # deg in RA
COOLDOWN = 4         # s between stripes (be kind to the server + machine)

TEMPLATE = (
    "SELECT ra, dec, w1mpro, w2mpro, w1cov, elat FROM catwise_2020 "
    "WHERE w1mpro > 9 AND w1mpro < 16.4 AND w1mpro - w2mpro >= 0.8 "
    "AND ra >= {lo} AND ra < {hi}"
)


def main():
    edges = list(range(0, 360, STRIPE))
    total = 0
    for lo in edges:
        hi = lo + STRIPE
        path = os.path.join(OUT, f"cw_ra{lo:03d}_{hi:03d}.csv")
        if os.path.exists(path) and os.path.getsize(path) > 100:
            n = sum(1 for _ in open(path)) - 1
            print(f"[skip] {lo:3d}-{hi:3d}  already have {n:,}", flush=True)
            total += n
            continue
        t = time.time()
        try:
            # ASYNC: sync queries time out on ~240k-row stripes; async has no such cap
            job = SVC.submit_job(TEMPLATE.format(lo=lo, hi=hi))
            job.run()
            job.wait(phases=["COMPLETED", "ERROR", "ABORTED"], timeout=1800)
            if job.phase != "COMPLETED":
                print(f"[FAIL] {lo}-{hi}: phase={job.phase}", flush=True); sys.exit(1)
            tab = job.fetch_result().to_table()
            try:
                job.delete()
            except Exception:
                pass
        except Exception as e:
            print(f"[FAIL] {lo}-{hi}: {str(e)[:160]}", flush=True)
            sys.exit(1)
        tab.write(path, format="ascii.csv", overwrite=True)
        total += len(tab)
        print(f"[ok]  {lo:3d}-{hi:3d}  {len(tab):>7,} rows in {time.time()-t:5.1f}s "
              f"-> {os.path.basename(path)}  (cum {total:,})", flush=True)
        time.sleep(COOLDOWN)
    print(f"\nDONE. total CatWISE quasar candidates (all-sky, pre-mask) = {total:,}")
    print("(Secrest+2021 final after |b|>30 + point masks = 1,355,352)")


if __name__ == "__main__":
    main()
