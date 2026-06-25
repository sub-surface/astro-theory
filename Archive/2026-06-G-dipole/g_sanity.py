#!/usr/bin/env python3
"""
G sanity check: can this machine recover a known dipole at CatWISE scale, cheaply?
Injects a dipole of known amplitude/direction into a Poisson HEALPix count map
(Nside=64, ~1.3M sources over |b|>30 sky), fits the Ellis-Baldwin least-squares
dipole, and checks recovery. Single-threaded, tiny memory. No healpy (astropy_healpix).
"""
import time, tracemalloc
import numpy as np
from astropy_healpix import HEALPix
from astropy.coordinates import Galactic
import astropy.units as u

rng = np.random.default_rng(42)
NSIDE = 64
N_SOURCES = 1_360_000          # CatWISE-like
D_TRUE = 0.0155                # CatWISE-like fractional dipole amplitude
L_TRUE, B_TRUE = 264.0, 48.0   # ~CMB direction (Galactic deg)

t0 = time.perf_counter(); tracemalloc.start()

hp = HEALPix(nside=NSIDE, frame=Galactic())
npix = hp.npix
# pixel centers -> unit vectors (the dipole design matrix)
lon, lat = hp.healpix_to_lonlat(np.arange(npix))
cb = np.cos(lat.rad)
nx = cb*np.cos(lon.rad); ny = cb*np.sin(lon.rad); nz = np.sin(lat.rad)

# galactic-plane mask |b|<30
mask = np.abs(lat.deg) >= 30.0
# true dipole direction unit vector
br, lr = np.deg2rad(B_TRUE), np.deg2rad(L_TRUE)
dvec = np.array([np.cos(br)*np.cos(lr), np.cos(br)*np.sin(lr), np.sin(br)])
dotp = nx*dvec[0] + ny*dvec[1] + nz*dvec[2]

# expected counts per pixel with dipole modulation, then Poisson-sample
nbar = N_SOURCES / mask.sum()
expected = nbar * (1.0 + D_TRUE*dotp) * mask
counts = rng.poisson(np.clip(expected, 0, None)).astype(float)

# --- least-squares dipole fit on unmasked pixels: n = A0 + A.(x,y,z) ---
m = mask
A = np.column_stack([np.ones(m.sum()), nx[m], ny[m], nz[m]])
coef, *_ = np.linalg.lstsq(A, counts[m], rcond=None)
A0, Ax, Ay, Az = coef
amp = np.sqrt(Ax**2+Ay**2+Az**2)/A0
dirvec = np.array([Ax,Ay,Az])/np.sqrt(Ax**2+Ay**2+Az**2)
b_fit = np.rad2deg(np.arcsin(dirvec[2]))
l_fit = np.rad2deg(np.arctan2(dirvec[1], dirvec[0])) % 360
sep = np.rad2deg(np.arccos(np.clip(dvec@dirvec, -1, 1)))

cur, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
dt = time.perf_counter()-t0
print(f"Nside={NSIDE}  npix={npix}  unmasked={m.sum()}  mean counts/pix={nbar:.1f}")
print(f"TRUE  D={D_TRUE:.4f}  (l,b)=({L_TRUE:.1f},{B_TRUE:.1f})")
print(f"FIT   D={amp:.4f}  (l,b)=({l_fit:.1f},{b_fit:.1f})  dir offset={sep:.2f} deg")
print(f"recovery: amp ratio={amp/D_TRUE:.3f}   |  wall={dt:.2f}s  peak_py_mem={peak/1e6:.1f} MB")
