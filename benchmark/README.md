# Independent OCR Benchmark v5 — original photos included

Install `benchmark/` as a directory at the root of the current project. **No working-project audit.** Do not import from `App/` for baseline tests. Optional GridMesh can be added later only after user approval.

Now included physically: three original JPEG photographs of actual Torgsoft receipts, ground-truth JSON and manual ROI for each; 1, 2, 8 product rows. Four previous draft GTs are retained separately without their missing original images. The 19-row receipt is excluded from the active dataset.

## Zero-inference preparation

```powershell
python -m benchmark validate
python -m benchmark plan
python -m unittest discover -s benchmark/tests -v
```

Only after express approval to load OCR models / run benchmarking:

```powershell
python -m benchmark run --confirm-load
python -m benchmark run --confirm-load --repeats 1   # smoke only
```

## Optional Grid Mesh + total (annotation-assisted)

```powershell
python -m benchmark run --confirm-load --crop-source gridmesh_total
python -m benchmark.tools.gridmesh_roi_report
```

This opt-in experiment imports `App.geometry`, retaining the independent ROI
baseline as the default. Product/VAT cells are matched using manual ROI centers;
ambiguous or missing matches fail explicitly without ROI fallback. The printed
total uses its manual annotation as a separate one-cell mesh with synthetic
rectangular boundaries. This is not automatic total detection or measured curved
footer rectification. No expected text or numeric value is used to generate crops.
Inverse coordinate maps are cached in `data/gridmesh_coordinates/`, keyed by
image pixels, annotation and cache version. Missing geometry is built once before
model loading and timed trials. Full receipt timing includes coordinate loading
and crop preparation; OCR-only timing excludes both. Changed image/annotations
produce a new cache entry. The manifest and HTML identify this mode.

The project `.venv` and third-party libraries are shared with the main project; no separate install is required when `requirements.txt` is already satisfied. Model binaries live in the shared project `models/` directory, not inside `benchmark/`. The third-party `v6_tiny_cyrillic_optional` ONNX is checked against its configured SHA256.

Local empirical timings only. `single_cell`: call recognizer separately for each physical ROI; `batch`: same ROI in groups. Default five repetitions; warm-up once per model. Numeric ROI is automatically tightened around printed digits; multi-line names are split into lines and joined with hyphenation repair. CPU/wall/RAM measured on local machine; no benchmarks copied from any source.

Active product names and numbers are agent-verified; owner approval remains pending. Numeric GT and names are marked agent-verified; `approved_by_user=false` remains until owner approval. For image previews with editable ROI use `benchmark/tools/annotate_roi.html`.
