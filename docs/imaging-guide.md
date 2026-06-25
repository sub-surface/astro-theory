# Imaging guide — picking the right survey for a position

How to turn `RA Dec` into a *good* image (and the right literature), instead of
defaulting to a low-res all-sky survey. Part of this is encoded in
[`celestrium/cutouts.py`](../celestrium/cutouts.py) (`identify_field`, `best_color_hips`,
`smart`); the judgment calls live here. Both Claude and Leon follow this when a
coordinate shows up — e.g. `213.6905918 -12.5801013`.

## The loop

```
RA, Dec ─▶ 1. resolve  ─▶ 2. classify ─▶ 3. choose survey ─▶ 4. set FOV ─▶ render
                │                                                   │
                └──────────────▶ 5. literature (what's known) ◀─────┘
```

### 1. Resolve — *is anything catalogued here?*
Cone-search SIMBAD (and NED for extragalactic) within ~1–2′:
`celestrium/cutouts.py:identify_field` returns `(main_id, object_type, separation″)`.
- **A hit** → you know what you're looking at; caption + scale the image to it.
- **No hit** → a "blank"/uncharted field (like 213.69 −12.58). Still imageable, but
  expect a sparse star field; the interesting move may be *why* nothing's catalogued.

### 2. Classify — the object type drives everything downstream
SIMBAD `otype` →
| type | examples | imaging implication |
|---|---|---|
| point-like | star, QSO, `Rad`, `X` | tight FOV (1–3′), high-res optical/IR |
| extended | galaxy `G`, `GiG`, nebula `Neb`, `SNR` | wide FOV (5–15′), deep optical colour |
| cluster | `ClG`, `GrG` | very wide FOV (15–30′), look at member distribution |
| blank | (no match) | default 5′; consider multi-λ to find faint/odd sources |

### 3. Choose survey — three axes

**(a) Declination footprint** (the hard constraint — a survey can't image sky it
never observed). Deep optical colour, best-first (encoded in `COLOR_FOOTPRINTS`):

| Survey | Dec coverage | depth / res | HiPS id |
|---|---|---|---|
| **Legacy Surveys DR10** | −68° … +84° | deep (r≈23), 0.26″/pix | `CDS/P/DESI-Legacy-Surveys/DR10/color` |
| **Pan-STARRS DR1** | > −30° | r≈23, 0.25″/pix | `CDS/P/PanSTARRS/DR1/color-z-zg-g` |
| **DES DR2** | < +5° (south) | deep, 0.26″/pix | `CDS/P/DES-DR2/ColorIRG` |
| **SDSS9** | N. galactic cap | r≈22 | `CDS/P/SDSS9/color` |
| **DSS2 colour** | all-sky | shallow, ~1″ | `CDS/P/DSS2/color` (fallback only) |

Rule of thumb: **Legacy DR10 unless it's not covered, then Pan-STARRS (north) /
DES (deep south), DSS2 only as last resort.** Near the Galactic plane all optical
surveys suffer extinction/crowding — note it, consider IR.

**(b) Wavelength of interest** — the multi-λ panel (`cutouts.panel`, via SkyView):

| regime | survey | sees |
|---|---|---|
| UV | GALEX NUV/FUV | star formation, hot stars, QSOs |
| optical | DSS2 / (Legacy via HiPS) | stars, galaxy morphology |
| near-IR | 2MASS J/H/K | cool/old stellar pops, dust penetration |
| mid-IR | WISE 3.4–22 µm | AGN, dust, debris discs (project B) |
| radio | NVSS / VLA FIRST / VLASS | AGN jets, synchrotron (dipole populations) |

For a known AGN, lead with radio + mid-IR; for a galaxy, optical colour + near-IR;
for a candidate IR-excess star (project B), WISE bands + optical for the host.

**(c) Resolution vs. field** — high-res (Pan-STARRS/Legacy/HST-via-MAST) for
morphology of a compact source; wide-shallow (DSS2/SkyView) for context and faint
extended emission. Don't pull HST/JWST pixels for a quick look — that's MAST + a
reason.

### 4. Set the FOV
Scale to the object's angular size: point source 1–3′, typical galaxy 5–10′,
nearby/large galaxy or cluster 15–30′. `smart()` picks 8′ for extended `otype`s,
3′ for point-like, 5′ for blank — override when you know the size (NED gives it).

### 5. Literature — *what's already known here*
- **SIMBAD bibliography** (`resolvers.bibliography`) — papers referencing the object.
- **NED** — extragalactic redshift, photometry, cross-IDs.
- **ADS/SciX** (`celestrium/ads.py`) — full search: by object name, or by coordinates
  (`object:"..."`), or topic; export to `refs.bib`. This is where a quick look turns
  into a literature trail.

## Worked examples
- **`187.7059 +12.3911`** → resolves to *M87* (`Rad`/galaxy), Dec +12 → Legacy DR10,
  FOV 3′ → a deep image of the giant elliptical + halo. ✓ what you'd want.
- **`213.6906 −12.5801`** → no SIMBAD object within 2′ → blank field; Legacy DR10
  still covers it, but expect a sparse star field. The multi-λ panel is the more
  informative product here (is there a faint radio/IR source the optical misses?).

## What's encoded vs. still manual
- **Encoded:** resolve, footprint-aware colour survey, FOV-by-type, multi-λ panel.
- **Still judgment:** picking the *scientifically* right band for the question,
  reading angular size from NED, deciding when a quick-look warrants real HST/JWST
  pixels. Extend `cutouts.py` opportunistically as patterns recur.
