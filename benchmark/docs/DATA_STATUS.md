# Data inventory (V5)

**Included, original JPEG bytes:**
- `1000696512.jpg`: one product, 55 564 ₸; GT + coordinates included; draft names.
- `1000697084.jpg`: two products, 51 504 ₸; GT + coordinates included; draft names.
- `1000696513.jpg`: nineteen products, 60 877 ₸; GT + coordinates included; draft names.

**Not included as pixels:** four earlier reference images (1, 3, 8, 11 products). Their draft GT are in `data/ground_truth_pending/` and are NOT counted as available. Do not claim that ZIP has their originals.

Medium-sized separate receipt still needed for a balanced short/medium/long selection. All numerical GT should be visually approved by user before reporting final numeric accuracy, despite passing arithmetic checks.

Image hashes and ROI fields are in `data/annotations/*.json`. `python -m benchmark validate` checks presence, hashes, math, and field IDs without running OCR.
