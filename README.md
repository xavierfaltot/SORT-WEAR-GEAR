# SORT WEAR GEAR v0.4 — MULTI-MATCH + REVIEW

Local-first visual sorting machine for generated fashion image libraries.

## Input

- `REFERENCES/PEOPLE/` — reference people / characters
- `REFERENCES/GEAR/` — exact garment references
- `TO_SORT/` — generated assemblies

The reference filename stem becomes the category name.

## Output

```text
REFERENCES/
├── PEOPLE/
└── GEAR/
TO_SORT/
SORTED/
├── BY_PEOPLE/
└── BY_GEAR/
REVIEW/
sorting_report.csv
library.json
predictions.json
corrections.json
```

Original images are never moved or deleted. `SORTED` and `REVIEW` contain generated copies and can be rebuilt safely.

## v0.4 vision architecture

```text
ASSEMBLAGE
  ↓
OpenCLIP ViT-B-32 semantic embedding
  +
DINOv2-small instance embedding
  ↓
PERSON consensus: full + upper + center
  ↓
GARMENT regions: upper + lower + feet + center + full
  ↓
MULTI-GARMENT selection
  ↓
AMBIGUITY thresholds / score margins
  ↓
REVIEW
```

This is intentionally different from a single semantic classifier: multiple body regions can nominate different exact garment references in the same image.

## REVIEW

The Review tab shows only ambiguous cases and lets you:

- inspect the predicted image;
- change the predicted PERSON;
- add or remove any GARMENT;
- inspect top candidates and scores;
- validate the result.

Validated corrections are saved to `corrections.json` and reused automatically on future scans of the same library.

## Install on macOS

```bash
cd SORT-WERA-GEAR
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

The first vision run downloads the OpenCLIP and DINOv2 model weights. After they are cached, matching runs locally on the Mac. Apple Silicon uses MPS automatically when available.

## Default thresholds

- PERSON minimum score: `0.42`
- GARMENT minimum score: `0.47`
- Maximum garments per image: `5`

These values are deliberately exposed in the UI because the best threshold depends on how visually similar the reference wardrobe is.

## Current limitation / next precision layer

v0.4 uses deterministic body-region crops rather than a garment segmentation model. This keeps the first robust build simple and local. If exact matching remains weak for near-identical garments, the next layer should be segmentation/object regions + DINO patch-level matching rather than lowering thresholds blindly.
