# AURORA — Antarctic Unified Routing & Operational Risk Analytics

**AURORA** is the umbrella product. This repository is its **route-optimization** codebase, and it
also ships the sea-ice-concentration (SIC) forecaster that feeds the router.

| AURORA component | Lives in | Status (verified 2026-09-28) |
|---|---|---|
| SIC forecasting | `backend/src`, `backend/scripts` (PolarPath ConvLSTM, 3-seed ensemble) | **Trained.** Checkpoints committed: `backend/runs/final_10ch_3f_seed{0,1,2}/best_model.pt` (1,083,833 B each) |
| Route optimization | `src/routing`, `src/ml/dynamic_reroute.py`, `backend/api/main.py` | **Implemented.** `POST /api/route/optimize`, `POST /api/route/reroute`, `GET /api/route`, safety/path validation included |
| SAR iceberg detection | separate repo `sakshidas1-ux/sar-iceberg-detection` | **PENDING / SEPARATE REPOSITORY** — no iceberg detections are produced from this repo |

Verified facts:

- `SIC MODEL IMPORT: PASS` — `sys.path.insert(0,'backend/src'); from model import ConvLSTMForecaster`
- `TEST DATASET IMPORT: PASS` — `from test_dataset import TestDataset`
- `python -m pytest tests/ -q` → **284 passed**
- Iceberg hook in this repo: `src/data/adapters.py::IcebergAdapter` (expects a caller-supplied
  risk field in [0,1]) and `GET /api/layers/status` reports `available: false`.
  `backend/cache/routing_metadata.json` marks `iceberg_standoff: "deferred - no trajectory
  predictor yet"`. The iceberg model itself lives in the separate repository above.

Known blocker (data, not code):

- `python backend/scripts/inference_2026.py` loads the ConvLSTM ensemble successfully but cannot
  run because `backend/data/test_2026/processed/sic.npy` (and the other forcing inputs) are absent
  on this machine. These files are **not** fabricated or generated; the run stays blocked until the
  real data is restored.

The rest of this document describes the SIC forecaster itself.

## Overview

A ConvLSTM ensemble for short-term Antarctic sea-ice concentration (SIC) forecasting in the Bharati–Maitri corridor of the Southern Ocean, built for NCPOR resupply-vessel route planning. The model ingests 5 days of satellite, atmospheric, and ocean data and predicts SIC at three horizons (day-1, day-2, day-3) over the Indian Ocean sector (10°W–80°E, 75°S–50°S) on a 0.25° regular grid. A three-seed ensemble with MC-Dropout quantifies uncertainty, which is converted into conformal 90% intervals calibrated once on 2025 and applied unchanged to the held-out 2026 test window.

The forecaster combines three public data sources — NSIDC CDR v6 satellite SIC, ERA5 atmospheric reanalysis, and CMEMS GLORYS12V1 ocean reanalysis — into a common 0.25° grid. It was trained on 2021–2024 (1,461 days), validated on 2025, and tested on an entirely held-out 2026 window (Jan 1 – Jun 23). The reference metric is the ratio of model MIZ RMSE to persistence MIZ RMSE; on day-1 that ratio is **0.69 in both the 2025 validation and the 2026 test years**, i.e. the model is ~31% better than persistence on day-1 and the skill transfers across years.

## Quickstart: running what ships in this repo

This branch ships the final trained models, the routing grids, the metrics, and all frontend
assets. The raw satellite/reanalysis downloads (~40 GB) and the regenerable ≥100 MB ensemble
cache are **not** committed (git ignores them — see the per-file size note in
`.gitignore`), so the router and the frontend run directly from the committed artifacts.

### 1. Interactive Full-Stack Web Application (Recommended for Hackathon Evaluation)

The Flask backend compositor serves both the complete REST API (27 endpoints) and the pre-built, production-ready React Aurora frontend (`frontend/aurora/dist`) in a **single command** — zero Node/npm setup required:

```bash
# 1. Activate your Python environment (Python 3.10+):
.venv\Scripts\activate            # Windows (or: source .venv/bin/activate on Linux/macOS)

# 2. Start the API & Application server:
python -m backend.api.main --port 8078
```

Open **`http://localhost:8078/`** in your browser. You can explore:
- **Route Planner (`/routes`)**: Interactive 8-connected A* environmental routing with real-time waypoint replanning, ice avoidance, and dynamic rerouting across Cape Town, Maitri, Bharati, and custom stations.
- **Operations Map (`/navigation`)**: High-fidelity Leaflet map with Web Mercator projected sea-ice rasters, Natural Earth coastlines, and Indian Antarctic stations.
- **SIC Forecaster (`/forecast`)**: 3-day lead-time ConvLSTM ensemble forecast player with uncertainty intervals and MIZ error diagnostics.
- **Analytics & Overview (`/analytics`, `/`)**: Comprehensive evaluation metrics, regret distributions, and model performance comparisons.

*(Optional)* For frontend development with live hot-reloading:
```bash
cd frontend/aurora
npm install
npm run dev      # Opens on http://localhost:5173 (automatically proxies /api to port 8078)
```

### 2. Static Lightweight Viewer (Zero-Dependency Fallback)

No build step and no backend process needed — `frontend/index.html` + `frontend/data/*` are pre-generated static files:

```bash
python -m http.server 8000 --directory frontend        # open http://localhost:8000/
```

### 3. Backend Route Engine & Verification

The A* router reads the committed grids in `backend/cache/` (`routing_sic_2026.npy`,
`routing_cost_x/y_2026.npy`, `routing_multiplier_2026.npy`, `routing_station_goals.json`, land/override
masks) and reproduces the validated Cape Town → Maitri route:

```bash
python backend/scripts/build_router.py
# -> outputs/final_demo/SIH2026059_final_route.png, backend/cache/route_capetown_maitri_2026-01-06.json
```

All reported numbers are already committed (`backend/cache/metrics_*.json`,
`uncertainty_stats.json`, `route_*.json`), so the backend analysis is verifiable immediately without any raw satellite data downloads.

> To regenerate predictions, metrics, or the frontend data from raw observations instead of the
> committed artifacts, follow **Full reproduction** below — that path downloads ~40 GB of data
> and rebuilds the excluded `ensemble_2026.npy` cache array.

## Results Summary

2025 values are loaded from `backend/cache/metrics_2025.json` / `backend/cache/metrics_3frame_2025.json` (mask-fixed 3-frame run); 2026 values from `backend/cache/metrics_2026.json`.

| Metric                     | 2025 (val) | 2026 (test) |
|----------------------------|-----------|-------------|
| MIZ RMSE day-1             | 0.0823    | 0.0958      |
| MIZ RMSE day-2             | 0.119     | 0.1265      |
| MIZ RMSE day-3             | 0.138     | 0.1447      |
| Persistence MIZ RMSE day-1 | 0.1187    | 0.1398      |
| **Ratio day-1**            | **0.69**  | **0.69**    |
| Ratio day-2                | 0.97      | 0.90        |
| Ratio day-3                | 1.11      | 1.03        |
| IIEE @ t=0.15 (aggregate)  | 0.034     | 0.073       |
| IIEE @ t=0.42 (aggregate)  | 0.026     | 0.058       |
| Coverage (90% conformal)   | 0.904     | 0.910       |
| Coverage (MIZ only)        | 0.959     | 0.880       |
| Samples                    | 360       | 167         |
| Evaluation window          | 2025 full | 2026 Jan–Jun|

Notes: 2025 day-2/day-3 model and persistence values (0.119/0.123, 0.138/0.124) are brief references reproduced as literals in `backend/scripts/gate5_2026.py`; the 2025 day-1 row is from `backend/cache/metrics_3frame_2025.json`. Coverage is computed with the frozen 2025 conformal quantiles; IIEE is cosine-latitude weighted.

Key findings:

- **Day-1 skill generalizes across years** — the persistence ratio is 0.69 on both 2025 (0.0823/0.1187) and 2026 (0.0958/0.1398).
- **Day-3 is worse than persistence in both years** (ratio 1.11, 1.03). This is reported honestly as a limitation; the product defaults to day-1/day-2.
- **Conformal intervals are calibrated** — 0.90 nominal → 0.904 (2025) and 0.910 (2026) observed all-valid coverage. MIZ-only coverage dropped 0.959 → 0.880 when the frozen quantiles meet a harder season (see Limitations).
- **IIEE rose 0.034 → 0.073, but absolute errors fell ~3×** (over-predictions 31,739 → 10,461; under-predictions 8,431 → 2,806 at t=0.15). The ratio rose because the Jan–Jun window has ~6.5× less ice-covered area than the full 2025 calendar year.

## Data

Three data sources, fused onto a common 101 × 361 (lat × lon) 0.25° grid:

| Source  | Product | Native resolution | Cadence | Role |
|---------|---------|-------------------|---------|------|
| NSIDC   | CDR v6 (G02202, NSIDC-0051) | 25 km polar stereographic | Daily | SIC target + input |
| ERA5    | Single-level reanalysis | 0.25° regular lat/lon | Hourly → daily mean | Atmospheric forcing (u10, v10, t2m) |
| CMEMS   | GLORYS12V1 | 1/12° regular lat/lon | Daily | Ocean forcing (uo, vo, thetao, so, zos; depth ~0.494 m) |

- **ROI:** lon −10°…80°E, lat −75°…−50°S (101 lat × 361 lon at 0.25°).
- **Training + validation record:** 2021-01-01 … 2025-12-31, 1,826 contiguous days (train ≤ 2024-12-31 = 1,461 days; 2025 = 365 validation days).
- **Test record (held out):** 2026-01-01 … 2026-06-23, 174 days → 167 windowed samples.

### Why this region

Covers both NCPOR year-round stations — Maitri (70.767°S, 11.733°E) and Bharati (69.397°S, 76.247°E) — and the ice-affected portion of the Cape Town → station transit corridor. The ROI contains 9 research stations from 7 nations (`frontend/data/stations.json`).

### Why this temporal split

2021–2024 spans three distinct ENSO regimes and the 2023 Antarctic sea-ice record minimum. 2025 is the validation year (model selection, early stopping, and conformal calibration). 2026 is fully held out — no calibration or model-selection decision used any 2026 value.

## Preprocessing Pipeline

Each source is regridded onto the ERA5 0.25° ROI grid (`backend/src/preprocess.py`):

| Source | Method | Notes |
|--------|--------|-------|
| NSIDC | pyresample `resample_nearest`, pyproj EPSG:3976 → EPSG:4326 | Curvilinear polar-stereographic source; radius of influence 50 km; fill NaN |
| CMEMS | xarray `.interp` linear, 1/12° → 0.25° | Regular grid; shallowest depth level (~0.494 m); thetao converted °C → K |
| ERA5 | Native grid, ROI crop | Already 0.25°; hourly → daily mean |

Time alignment is by **calendar date**, not array index; any date missing from any source is dropped. The common intersection is asserted gapless (`np.diff(...) == 1 day`). CMEMS duplicated boundary timestamps are removed by timestamp before regridding.

### Input channels (10)

| # | Channel | Source | Standardization |
|---|---------|--------|-----------------|
| 0 | SIC | NSIDC | standardized on the fly with train scalars (`sic_mean.npy`, `sic_std.npy`) |
| 1 | u10 | ERA5 | standardized |
| 2 | v10 | ERA5 | standardized |
| 3 | t2m | ERA5 | standardized |
| 4 | uo | CMEMS | standardized |
| 5 | vo | CMEMS | standardized |
| 6 | thetao | CMEMS | standardized |
| 7 | so | CMEMS | standardized |
| 8 | zos | CMEMS | standardized |
| 9 | SIC_prev_year | NSIDC (shifted −365 days) | standardized with the SIC scalars |

The SIC target stays **raw 0–1** (not standardized); the sigmoid output head matches that range directly. Forcing channels 1–8 are standardized in `preprocess.py` with train-period stats; SIC_prev_year is built in `preprocess.py` by strict calendar shift (t − 365 days). For 2021 the lookup (2020 not on disk) is NaN → 0 post-standardization (365 days, `norm_stats.json`). For 2026 the lookup uses real 2025 SIC.

### Cached arrays (`backend/data/processed/`)

| File | Shape | Contents |
|------|-------|----------|
| `sic.npy` | `[1826, 101, 361]` | Raw SIC 0–1, NaN preserved |
| `forcing.npy` | `[1826, 9, 101, 361]` | Standardized forcing (channels 1–9), NaN → 0 |
| `time.npy` | `[1826]` | Calendar dates |
| `lat.npy` / `lon.npy` | `[101]` / `[361]` | −75…−50 / −10…80 |
| `sic_mean.npy`, `sic_std.npy` | scalar | Train-period SIC stats |
| `norm_stats.json` | — | Per-channel mean/std, training period, channel names |

Test arrays live in `backend/data/test_2026/processed/` (`sic.npy` `[174,101,361]`, `forcing.npy` `[174,9,101,361]`).

## Model Architecture

Two-layer ConvLSTM with a 3-frame sigmoid head (`backend/src/model.py`; `ConvLSTMCell` written from scratch):

```
Input:  [B, 5, 10, H, W]
        │
        ▼
ConvLSTMCell(10 → 32), kernel 3, padding 1
        │
        ▼
ConvLSTMCell(32 → 64), kernel 3, padding 1
        │
        ▼
Dropout2d(p=0.1)      ← MC Dropout at inference
        │
        ▼
Conv2d(64 → 3, kernel 1) + sigmoid
        │
        ▼
Output: [B, 3, H, W]  (SIC at D+1, D+2, D+3)
```

| Property | Value |
|----------|-------|
| Total parameters | 270,147 (asserted in `model.py` self-check) |
| Lookback / targets | 5 input days → 3 target days |
| Dropout | p=0.1, applied before the head |
| Output activation | Sigmoid (bounded [0, 1]) |
| Peak VRAM guidance | < 6 GB at batch 4 (CHECK 4 dry-run criterion in `train.py`) |

`ConvLSTMCell` uses the standard update: gates from one convolution of `concat(input, hidden)`, chunked into input/forget/candidate/output; `c_t = f⊙c + i⊙g`; `h_t = o⊙tanh(c_t)`.

## Training

Final models were trained with `backend/src/run_final.py` (the 3-frame script, not the legacy 1-frame `train.py`).

| Hyperparameter | Value |
|---------------|-------|
| Batch size | 4 |
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| Scheduler | CosineAnnealingLR, T_max = 100 |
| Max epochs | 100 |
| Early-stop patience | 10 |
| Early-stop / checkpoint metric | Validation MIZ RMSE, mean of the 3 horizons (`miz_mean`) |
| Random seeds | 0, 1, 2 (one model per seed) |

### Loss function

Horizon-weighted masked MSE over the three target days:

```
L = Σ_h w_h · masked_MSE(pred_h, target_h, mask_h) / Σ w      w = [3.0, 1.5, 0.5]
masked_MSE:  pixel weight = 3.0 where 0.15 < target < 0.85 (MIZ emphasis), else 1.0;
             then multiplied by the valid mask (NaN cells excluded)
```

### Ensemble

Three models trained from scratch with seeds 0, 1, 2. At inference each model runs 20 MC-Dropout passes; the ensemble prediction is the mean of the three per-model means, and uncertainty combines ensemble spread with MC variance:

```
combined_std = sqrt(ens_std² + mc_std_avg²)     (eval.py:89, inference_2026.py:161)
```

All-pooled 5/50/95 quantiles are also computed per sample. Confidence classes follow `combined_std`, binned over **ice cells only** (valid cells where predicted SIC > 0.15): `std` < p33 → HIGH, p33 ≤ `std` < p66 → MEDIUM, `std` ≥ p66 → LOW, invalid = 255 (`backend/cache/confidence_class_*`). p33/p66 are quantiles of the pooled 2026 test distribution, written to `frontend/data/confidence_thresholds.json` — the single source of truth the frontend loads at runtime (not hardcoded). This is the third iteration of the bins: fixed 0.05/0.10 thresholds (99.75% of cells HIGH), then p50/p90 quantiles over all valid cells (dominated by open water, so labels encoded "has ice"), now p33/p66 over ice cells so the label discriminates order within the ice pack (ice interior → HIGH, MIZ → MEDIUM/LOW). Open-ocean cells, having tiny `combined_std`, naturally fall in HIGH.

Measured wall time: seed 0 ran 24 epochs in 2,313 s (≈39 min), best epoch 15 (`backend/runs/final_10ch_3f_seed0/metrics.json`).

## Evaluation

### Metrics

| Metric | Definition |
|--------|-----------|
| **MIZ RMSE** | Pooled RMSE over valid cells where 0.15 < true SIC < 0.85 |
| **Persistence MIZ RMSE** | Same, using the observed SIC k days before horizon k (day-1 = last input day; day-2/3 = observed day-1/day-2) |
| **Ratio** | Model MIZ RMSE / persistence MIZ RMSE |
| **IIEE** | (over + under) / true ice area, cosine-latitude weighted; reported as aggregate (ratio of weighted totals) |
| **Coverage** | Fraction of valid cells where true SIC ∈ [median − q90, median + q90] |

All metrics use `backend/cache/valid_mask.npy` (26,498 of 36,461 cells = 72.7% — the 9,963 cells never valid during training are excluded).

### Conformal calibration

The reported 90% intervals are **not** the raw `combined_std`. They are 90th-percentile residuals fit on a calibration half of 2025 (Jan 1 – Jun 30, 176 samples) and validated on the second half (Jul–Dec, 184 samples), stratified by true SIC:

```
if true_SIC > 0.15:  q90 = q90_miz   (0.12617)
else:                q90 = q90_open  (0.00206)
interval = [clip(median − q90, 0, 1), clip(median + q90, 0, 1)]
```

Quantiles from `backend/cache/metrics_2025.json` (conformal widths: MIZ 0.2523, open 0.0041). The two-strata split is justified by the spread difference: MIZ-cell ensemble std (~0.021 mean) is ~4.5× the all-cell mean std (~0.0047) (`backend/cache/uncertainty_stats.json`). The 2025-fitted quantiles are **frozen** and applied unchanged to 2026 — refitting on test data would invalidate the coverage guarantee. 2025 coverage: 0.904 all / 0.959 MIZ; 2026 (frozen): 0.910 all / 0.880 MIZ.

### Threshold design (two-threshold product)

| Purpose | Threshold | Rationale |
|---------|-----------|-----------|
| Routing (product default) | 0.15 | Minimizes missed ice (under-prediction): 8,431 missed cells at t=0.15 vs 11,215 at t=0.42 (2025). Missed ice is the safety-critical error for navigation. |
| Reporting / metrics | 0.42 | Minimizes IIEE (0.026 vs 0.034 aggregate, 2025). Equal over/under weighting is appropriate for a single scalar comparison. |
| Captain's heatmap | Continuous SIC | No threshold applied |

Rationale text from `backend/cache/metrics_2025.json`. 2026 numbers: over/under = 10,461/2,806 at t=0.15 and 4,912/4,232 at t=0.42.

## Interactive Frontend

A self-contained static viewer (`frontend/`) for exploring predictions — no backend, no browser inference. Serve and open it:

    cd frontend
    python -m http.server 8000

(First load needs internet for D3 from the d3js.org CDN; hard-refresh with Ctrl+Shift+R after regenerating frames.)

`index.html` is a build artifact: `backend/scripts/write_html.py` regenerates it from `backend/scripts/template.html`, inlining dates, stations, coastline, and horizon metrics. At runtime the page fetches only the lat/lon grids and the per-date cell values.

### What you see

- 30 dates from the 2026 test window (Jan–Jun) × Day 1/2/3 horizons, playable at 4 fps.
- **Actual** (green) and **Predicted** (orange) SIC maps side-by-side or overlaid with independent opacity sliders. The overlay blend is a visual diff: navy = open water, yellow/olive = both agree on ice, green alone = miss, orange alone = over-prediction.
- Toggleable layers: Stations, Grid, Actual SIC, Predicted SIC.
- Hovering any cell shows lat/lon, date + horizon, actual and predicted SIC, interval half-width at 90%, and confidence class + raw `combined_std` against the ice-cell p33/p66 range (HIGH/MEDIUM/LOW). The thresholds are loaded at runtime from `frontend/data/confidence_thresholds.json`, not hardcoded. The confidence label is a self-assessed ensemble spread and is distinct from the calibrated 90% interval, which stays a separate row. Land, ice shelf, and the coastal NaN band render transparent (tooltip shows coordinates only).

### Data & stack

`frontend/data/` — 360 pre-rendered frames (30 × 3 × actual/predicted/diff/uncertainty), 30 per-date cell-value JSONs + `date_index.json`, lat/lon grids, `coastline.json`/`stations.json`, `horizon_metrics.json`/`metrics.json`, and `confidence_thresholds.json` (ice-cell p33/p66 bins; the only confidence thresholds the page uses). No `manifest.json` or `.geojson` — overlays are plain JSON inlined at build time.

Stack: inline SVG + D3 v7 (the only external runtime dependency, from CDN); frames rendered offline with matplotlib; serving via `python -m http.server`.

**Colours** (defined in `backend/scripts/prep_frontend_data.py`): actual `#1B5E20`→`#E8F5E9` green, predicted `#BF360C`→`#FFFDE7` orange; open water is transparent so the navy base shows.

## Limitations & Honest Findings

- **Day-3 is worse than persistence** in both years (ratio 1.11 / 1.03); day-2 is at best parity (0.97 / 0.90). The product defaults to day-1; day-3 is provided for research only.
- **Reanalysis latency is not simulated.** Training uses same-day ERA5; ERA5 single-levels lag real time by days, so operational use requires NRT forcing and re-testing. This is documented here, not solved.
- **The 2026 MIZ-only coverage dropped 0.959 → 0.880.** The frozen 2025 quantile under-covers the harder 2026 season. All-valid coverage stays calibrated (0.910); 0.88 is documented as a known cost of freezing conformal quantiles across years.
- **The 2026 IIEE ratio (0.073) is larger than 2025 (0.034),** but absolute weighted error counts fell ~3× (over 31,739 → 10,461, under 8,431 → 2,806 at t=0.15). The ratio is a denominator artifact of the shorter, lower-ice Jan–Jun window — not a model regression.
- **Coastal NaN band.** NSIDC CDR v6 flags near-coast cells invalid (passive-microwave land contamination), so the model has no training signal within ~50–100 km of the coastline. The frontend renders these transparently.
- **Resolution is 25 km.** Leads (1–10 km scale) and fine marginal-ice-zone structure are not resolved; VIIRS-scale (750 m) data is future work.
- **Single region.** Trained on 10°W–80°E only; other sectors (Weddell, Ross) have different regimes and would require retraining.
- **Land-cell encoding.** ~27% of input SIC pixels are exact zeros (land/pole-hole fill). The loss correctly excludes these targets, but the convolution still spreads zero-value land features into receptive fields of valid edge cells. There is no explicit "no data" input signal.

## What We Tried and Did Not Use

This section documents experiments and design alternatives explored during development that were not adopted in the final model. Each entry states what was tried, the measured outcome where an artifact survives, and why the approach was set aside. Entries marked *historical record* rest on the development conversation record — the underlying files no longer survive on disk, so numeric claims are limited to what is still documented elsewhere.

### Architecture experiments

**Light ConvUNet (abandoned).**
A ConvLSTM encoder/bottleneck/decoder with skip connections was implemented (`backend/runs/_archive/model_v2.py`) and trained for 23 epochs (`training_log_convunet.csv`). The encoder used the same ConvLSTMCell as the final model, with batch norm and Dropout(0.2) throughout and concatenated skip inputs in the decoder. At 1,155,939 parameters (~4.3× the final 270,147) it reached day-1 MIZ RMSE ≈ 0.088 (best 0.0875 at epoch 17), worse than the adopted ensemble's 0.0823. Epoch time averaged ~324 s vs ~50 s for the plain ConvLSTM (`training_log_unweighted.csv`), with higher peak memory. The skip-connection gains did not offset the complexity; reverted.

**Binary cross-entropy loss term (abandoned).**
The 1-frame predecessor (`backend/runs/_archive/run_final_bce.py`, and the legacy `backend/src/train.py`) added a BCE term at coefficient 0.5 to the MSE to sharpen the ice/no-ice decision boundary. BCE operates on a different magnitude than MSE, so the BCE term dominated the composite loss and the continuous SIC output collapsed toward the binarized target. The retrain's cached day-1 MIZ RMSE regressed to 0.161 against 0.123 persistence (ratio 1.30, `backend/runs/_archive/final_10ch_1f_bce_seed0/metrics.json`) — worse than persistence. Discarded; the final 3-frame loss is masked MSE alone.

**NIIEE as a loss term (abandoned).** *historical record*
Normalized Integrated Ice Edge Error (NIIEE) was initially included in the training loss because it is the metric the product ultimately reports. As implemented against a hard threshold (pred > 0.15) it has near-zero gradient almost everywhere, contributing a constant to the loss without a learning signal. The loss was reduced to masked MSE alone. NIIEE/IIEE is still computed as an evaluation metric (`backend/src/eval.py`), where the piecewise nature is not an issue.

**Equal-weighted multi-horizon loss (replaced).**
The 3-horizon loss originally weighted day-1/2/3 equally (archived unweighted checkpoint and log in `backend/runs/_archive/`). Equal weighting diluted the day-1 gradient — day-1 carries the cleanest signal while day-3 is noisier. The shipped weights are [3.0, 1.5, 0.5] for [day-1, day-2, day-3] (`backend/src/run_final.py`), normalized by their sum so the effective day-1 share of the loss is 0.6. The unweighted variant's artifacts were archived rather than deleted.

**Dropout rate 0.2 (reduced to 0.1).**
The first ConvLSTM used Dropout2d(p=0.2) (as did the ConvUNet). With only 1,461 training days this over-regularized and stalled validation loss early in both runs; `backend/src/model.py` documents the reduction to p=0.1 to avoid over-regularization on the small dataset. The retuned rate restored convergence, and MC Dropout at inference is preserved.

### Feature engineering

**Derived features: ice-edge distance, SIC gradient, wind-stress curl (dropped).** *historical record*
Three derived channels were proposed to help the model focus on the marginal ice zone. None had a direct literature citation as an ML input for SIC forecasting, so they were dropped in favour of channels with published justification. Revisiting any of them later is cheap — they require only new preprocessing columns and a channel-count change in `backend/src/dataset.py`.

**ERA5 sea surface temperature (dropped).** *historical record*
`sea_surface_temperature` was considered as an additional channel. CMEMS `thetao` at ~0.494 m depth is effectively bulk SST and covers the same physical quantity, so carrying both was judged to add no value. ERA5 `sst` would also inherit the same ~5-day operational latency as the other single-levels, so it would not reduce the deployment gap.

**SIC from the previous year (kept).**
The one derived feature adopted: SIC at D−365 as channel 9, an annual-memory signal broadly inspired by temporal-weighting concepts in the SIC-forecasting literature (the DB-SICNet reference is from the design discussion). It is built in `backend/src/preprocess.py` and stored as the ninth forcing channel, standardized with the SIC scalars. The days before a full one-year lookback exists (~365) are filled and recorded as `sic_prev_year_nan_days` in `backend/data/processed/norm_stats.json`.

### Evaluation and calibration experiments

**Threshold sweep to reduce IIEE (misdiagnosed).**
An early attempt to fix a high IIEE score (~3.25 diagnostic total retained in `backend/cache/metrics_3frame_2025.json`) swept decision thresholds 0.15–0.54 (`backend/scripts/iiee_sweep.py`, `backend/scripts/diagnose_3f.py`). IIEE dropped below 1.0 at higher thresholds — `diagnose_3f.py` literally concludes "IIEE < 1.0? YES — threshold artifact, model is usable" — but the underlying cause was an evaluation-mask bug: the pipeline was scoring cells that were NaN throughout training. After the mask fix (`backend/scripts/mask_fix_3f.py`, 9,963 cells excluded), IIEE at t=0.15 fell to 0.034 (`backend/cache/metrics_2025.json`) with no model change. The sweep was abandoned.

**Isotonic recalibration (not applied).** *historical record*
When a reliability check suggested over-prediction in open-ocean cells, isotonic regression was proposed to correct the bias. The apparent bias was the same mask bug; after the fix the open-ocean mean prediction is 0.005 (`backend/cache/metrics_3frame_2025.json`). Recalibrating would have papered over a bug, so it was not applied.

**Safety-conservatism framing (withdrawn).** *historical record*
An earlier framing described the over-prediction as deliberate safety conservatism for routing. The reliability data showed a systematic bias, not a policy choice, so the framing was withdrawn and the over-prediction is documented as a limitation (see above) instead of a feature.

**Confidence scalar formula (removed).** *historical record*
An exponential formula was proposed to collapse uncertainty into a single "confidence" number; its divisor was not derived from any measurement. It was removed in favour of reporting the raw `combined_std` and letting the frontend bin it by empirical quantiles (`backend/scripts/prep_frontend_data.py`).

**Pooled conformal quantile (rejected).**
A single shared q90 was considered instead of separate MIZ/open-ocean quantiles. The spread difference on disk justifies stratification: MIZ-cell ensemble std (mean 0.021) is ~4.5× the all-cell mean std (0.005) (`backend/cache/uncertainty_stats.json`), and the fitted interval widths are 0.2523 (MIZ) vs 0.0041 (open water) (`backend/cache/metrics_2025.json`). A pooled quantile would over-widen open-ocean intervals and under-cover the MIZ; stratified quantiles were retained.

### Scope decisions

**VIIRS 750 m sea-ice concentration (out of scope).** *historical record*
The NOAA-20 VIIRS product provides SIC at ~750 m — the resolution a routing product would want — but it is optical/infrared and cloud-gapped, requiring a fusion pipeline with passive-microwave data (estimated ~2 weeks of work). The shipped product's 25 km resolution and its implications are documented as a limitation (see above). Deferred as future work.

**Iceberg drift model (out of scope).** *historical record*
The original problem statement mentions iceberg trajectory prediction. A physics-informed drift model would be a separate sub-project (historical tracks, force balance, residual-ML corrector) and was not built. Adding it would also change the model's output contract (currently B·3·H·W SIC only) and its evaluation.

**Route optimization (out of scope).** *historical record — superseded*
No NSGA-III, A*, or cost layer. The SIC forecast is the upstream input such a routing layer would consume; nothing in `src` or `scripts` emits a recommended trajectory or consumes a cost layer. The routing layer itself is not part of this submission.

> **Update:** this note described the earlier SIC-only submission. Route optimization *is* now
> implemented in this repository (`src/routing/`, `src/ml/dynamic_reroute.py`,
> `backend/api/main.py`), and the SIC ensemble is its input — see the status table at the top.

**ERA5 latency simulation (not implemented).**
In deployment ERA5 lags real time by ~5 days; training uses observed ERA5 aligned to input days — standard supervised practice but not operationally reproducible as-is. Simulating the gap by shifting ERA5 inputs and retraining is documented (see Limitations) but was not done.

**Physics-informed loss (deferred).** *historical record*
Temporal-smoothness and physical-bounds regularizers were considered. Bounds are already satisfied by the sigmoid output head (`backend/src/model.py`); a smoothness term risks suppressing real rapid ice change during storms. Both deferred.

### Frontend experiments

**PIL-based rendering (replaced by matplotlib).** *historical record*
When matplotlib's `_image` DLL was blocked by Windows security policy, PIL was used as a stopgap for QA plots. After the issue was resolved, matplotlib was restored for all rendering — the colormap, alpha transparency, and pixel-exact output required it — and remains the renderer in `backend/scripts/prep_frontend_data.py`.

**CesiumJS 3D globe (rejected).** *historical record*
CesiumJS was considered for the 3D polar view; the Cesium project does not support polar stereographic projection. deck.gl GlobeView was evaluated in turn but rejected in favour of the 2D frame-based viewer that ships with the repo.

## Future Scope

The current model provides short-term SIC forecasts for the Bharati–Maitri corridor. Several
extensions would materially improve its operational value. Each entry below is scoped,
data-backed, and achievable with a reasonable engineering effort.

### 1. Ice thickness overlay from USNIC Antarctic ice charts

The model currently predicts SIC (concentration) but not ice thickness. For ship routing,
thickness is equally important: two regions with identical SIC can differ by 60× in actual
thickness, determining whether a vessel can pass.

The U.S. National Ice Center (USNIC) produces Antarctic sea-ice charts that include
stage-of-development / estimated ice thickness information as defined by the World
Meteorological Organization (WMO). These charts are available in two forms:

| Product | Format | Access | Status |
|---------|--------|--------|--------|
| USNIC Arctic and Antarctic Sea Ice Charts (G10013) | SIGRID-3 vector shapefiles | NSIDC | Antarctic production suspended June 2023 |
| USNIC Sea Ice Concentration and Climatologies (G10033) | Gridded NetCDF | NSIDC (direct HTTP) | Available; includes thin, first-year, and multiyear ice fractions |
| Antarctic Stage of Development Chart | PNG graphical | USNIC website | Updated every other week (Friday) |

**How to add it:** Use G10033 for gridded fields. The dataset contains total ice concentration,
multiyear ice concentration, first-year ice concentration, thin ice concentration, and fast ice
extent. These are thickness **proxies** (not actual meters), but they map directly to the WMO
ice categories that ship operators already use in egg-code interpretation.

Download is a direct HTTP fetch from `https://noaadata.apps.nsidc.org/NOAA/G10033/` — no
authentication required. Gridded files are NetCDF, so they can be regridded onto the existing
0.25° ERA5 reference grid using the same pipeline as NSIDC CDR v6.

**Frontend integration:** Add a "Thickness" toggle layer that renders the thin/multiyear
fraction as a secondary colour overlay or as a distinct hatched pattern. Cell tooltips can
report "Thin ice: 60%, Multiyear: 15%" alongside the existing SIC value.

**Effort estimate:** ~1–2 days for data pipeline integration (download, regrid, cache), plus ~1
day for the frontend overlay.

**Caveats:** USNIC charts are bi-weekly, not daily. They carry a ~5-day observational lag. They
are analyst-drawn, not satellite-derived, so the spatial resolution is regional rather than
pixel-level. Treat the overlay as qualitative context, not as a quantitative thickness field.

### 2. Dynamic data fetching and continuous inference

The current pipeline is batch-oriented: preprocess, then infer, then stop. An operational
deployment needs to run continuously, fetching new data as it becomes available and producing
updated forecasts on a rolling basis.

**Data source cadences for operational use:**

| Source | NRT product | Update frequency | Latency |
|--------|------------|------------------|---------|
| NSIDC SIC | NRT CDR (G10016) | Daily | ~1 day |
| ERA5 atmosphere | ERA5T | Daily | ~5 days |
| CMEMS ocean | NRT + forecast (ANFC) | Daily | ~1 day |

**How to build it:** A simple scheduler (cron, GitHub Actions schedule, or a lightweight Python
loop) polls each source for new granules. When a new date is available across all three sources,
the scheduler:

1. Downloads the new files
2. Appends them to the existing processed arrays
3. Runs inference on the new sample
4. Updates the cache files that the frontend reads
5. Pushes updated data to the frontend

The frontend needs a WebSocket or polling mechanism to pick up new predictions without a manual
refresh.

**Key design decision:** The model weights are frozen. This is pure inference — no retraining.
The scheduler only extends the input arrays and re-runs the forward pass.

**Effort estimate:** ~3–5 days for the scheduler, incremental preprocessing logic, and frontend
polling.

**Caveats:** ERA5 latency means the atmospheric forcing will always be ~5 days stale in
operational use. This is a fundamental limitation of using ERA5 rather than a real-time NWP
product. For true operational deployment, replace ERA5 with ECMWF HRES or GFS forecast fields.

### 3. React + deck.gl frontend for 3D polar visualization

The current frontend is a 2D SVG + D3 viewer. It is functional and fast, but it does not convey
the polar geography naturally — the ROI appears as a wide strip rather than as part of a
spherical Earth.

**Why deck.gl specifically:** A 2024 evaluation by Development Seed tested three web mapping
libraries for visualizing data near the poles (Mapbox Globe View, MapLibre Globe View, and
deck.gl). Their conclusion: deck.gl was **the only library that provided an adequate solution**
for visualizing data close to the Earth's poles. Mapbox and MapLibre both have visible rendering
artifacts near the poles because their Web Mercator base tiles do not extend past ±85° latitude.

**Technical details:**

- deck.gl's `GlobeView` renders the earth as a 3D sphere and correctly projects data near the
  poles. A dedicated polar projection feature was added to the `RasterReprojector` module
  (developmentseed/deck.gl-raster PR #270) that handles polar tiles (|lat| > 75°) and the
  antimeridian crossing.

- `BitmapLayer` renders raster SIC fields onto the globe. The bitmap is projected onto the
  sphere via the `GlobeView` camera.

- `PathLayer` can render vessel routes as great-circle arcs along the sphere.

- `IconLayer` can place station markers with depth sorting so that markers behind the globe are
  occluded correctly.

**What the React version would add over the current frontend:**

| Feature | Current (SVG + D3) | React + deck.gl |
|---------|-------------------|-----------------|
| View | 2D flat map | 3D rotatable globe |
| Polar accuracy | Geometrically honest but non-spherical | Spherical; no polar distortion |
| Route rendering | Not implemented | Great-circle paths along the globe |
| Layer compositing | PNG overlays | GPU-composited WebGL layers |
| Performance at scale | Fine for 30 dates | Scales to thousands of frames |
| Ecosystem | Custom D3 code | Standard React component model |

**Effort estimate:** ~1–2 weeks for a full rebuild. The data layer (cache files, GeoJSON) stays
the same; only the rendering layer changes.

**Why it was not built for v1:** The 2D SVG frontend was sufficient for a 30-date demo and could
be built in 2 days. The 3D globe is a significant engineering effort that adds visual polish but
does not change the underlying model accuracy. It is the right choice for a production
deployment, not for a prototype.

### 4. Other directions

- **VIIRS 750 m SIC fusion.** The NSIDC CDR v6 product is at 25 km. NOAA-20 VIIRS provides SIC
  at ~750 m but is cloud-gapped. A fusion pipeline that merges VIIRS (clear-sky, high-res) with
  passive microwave (all-weather, low-res) would give the model access to ice leads and fine
  marginal-ice-zone structure. This is a 2-week data-engineering project.

- **Multi-region training.** The current model covers 10°W–80°E. Extending to the Ross Sea
  (160°E–150°W) and the central Weddell Sea (60°W–20°W) would require new downloads, new
  preprocessing, and retraining. The atmospheric and oceanic regimes differ, so a single model
  may not transfer — regional fine-tuning would be needed.

- **Latency-aware retraining.** Simulate the ERA5 5-day lag during training by shifting the
  atmospheric inputs backward in time. This would make the model operationally realistic without
  requiring a different data source.

- **Iceberg drift module.** The original problem statement mentions iceberg trajectory. A
  separate model — physics-based drift plus a learned residual correction — would be a distinct
  sub-project requiring historical iceberg tracks from the BYU/NIC database.

## Project Structure

```
sic/
├── README.md
├── requirements.txt
├── .gitignore
├── backend/
│   ├── src/                  # Training / evaluation code
│   │   ├── preprocess.py     # Align NSIDC + ERA5 + CMEMS → backend/data/processed
│   │   ├── dataset.py        # SICDataset (train/val sliding windows)
│   │   ├── test_dataset.py   # TestDataset (2026, 174 days → 167 samples)
│   │   ├── model.py          # ConvLSTMForecaster (from-scratch ConvLSTMCell)
│   │   ├── train.py          # Legacy 1-day trainer (debug / dry-run checks)
│   │   ├── run_final.py      # Final 3-frame trainer (seeds 0/1/2)
│   │   └── eval.py           # 2025 eval → backend/cache/metrics_2025.json, conformal split
│   ├── scripts/              # One-off pipelines: preprocess_2026.py, inference_2026.py,
│   │                         #   gate5_2026.py, conformal_3f.py, mask_fix_3f.py,
│   │                         #   diagnose_3f.py, iiee_sweep.py, iiee_invariant_check.py,
│   │                         #   prep_frontend_data.py, gen_metrics.py, write_html.py,
│   │                         #   template.html, qa_plot.py, report.py, find_nsidc_2026.py
│   ├── data/
│   │   ├── raw/              # NSIDC CDR v6, ERA5, CMEMS downloads (gitignored)
│   │   ├── processed/        # sic/forcing/time/lat/lon .npy + sic_mean/std + norm_stats.json
│   │   └── test_2026/        # Held-out 2026 arrays [174, ...]
│   ├── runs/
│   │   ├── final_10ch_3f_seed{0,1,2}/  # best_model.pt, metrics.json, training_log.csv
│   │   └── _archive/         # Rejected/legacy experiments (ConvUNet, BCE, unweighted, ...)
│   ├── cache/                # ensemble_*/true_*/uncertainty_*.npy, metrics_*.json,
│   │                         #   dates_2026.npy, valid_mask.npy, uncertainty_stats.json
│   └── plots/                # pred/true/dates .npy + metrics.json (2025 & 2026)
└── frontend/                 # Interactive map viewer (built by backend/scripts/write_html.py)
    ├── index.html            # built artifact — do not edit directly
    ├── data/                 # frames/, values/, stations.json, coastline.json,
    │                         #   horizon_metrics.json, metrics.json, lat/lon.json,
    │                         #   valid_mask.npy, valid_mask.png
    ├── _archive_2025/        # previous viewer generation (superseded)
    └── _archive_singlehorizon/
```

## Reproducing

> **Heads-up:** steps 2–6 below re-download the raw data (~40 GB), rebuild `backend/data/*`,
> and regenerate the ≥100 MB cache arrays that are gitignored on this branch — they need NSIDC
> Earthdata credentials, a `~/.cdsapirc` (ERA5), and a Copernicus Marine account. If you only
> want to run the shipped models, router, and frontend, use the **Quickstart** section above
> (steps 1 and the frontend serve also work from a fresh clone unchanged).

### 1. Environment

```
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

CUDA build used here: `torch 2.14.0+cu126` (cuda 12.6), Python 3.14. Use the GPU build of PyTorch so `.venv\Scripts\python` picks up CUDA.

### 2. Data

- **NSIDC CDR v6:** place daily `sic_pss25_*.nc` files in `backend/data/raw/nsidc/` (Earthdata download; `~/.netrc` / `_netrc` credentials for earthaccess).
- **ERA5:** `backend/src/preprocess.py` auto-downloads missing monthly NetCDFs via `cdsapi` using `~/.cdsapirc`.
- **CMEMS GLORYS12V1:** place `Ocean_YYYY_MM.nc` files in `backend/data/raw/cmems/` (Copernicus Marine account).

### 3. Preprocessing

```
python backend/src/preprocess.py            # 2021–2025 alignment → backend/data/processed
python backend/scripts/preprocess_2026.py   # 2026 test arrays → backend/data/test_2026/processed
```

Check `TEST_MODE=False` and `INCLUDE_CMEMS=True` at the top of `backend/src/preprocess.py`.

### 4. Training (3-frame ensemble)

```
python backend/src/run_final.py --seed 0
python backend/src/run_final.py --seed 1
python backend/src/run_final.py --seed 2
```

Each seed saves to `backend/runs/final_10ch_3f_seed{N}/` (seed 0 log records 24 epochs, 2,313 s wall time).

### 5. Evaluation (2025)

```
python backend/src/eval.py                 # → backend/cache/ensemble_2025.npy, metrics_2025.json, backend/plots/*
python backend/scripts/conformal_3f.py     # → backend/cache/ensemble_3frame_2025_*.npy, metrics_3frame_2025.json
```

### 6. Test (2026, held out)

```
python backend/scripts/inference_2026.py   # → backend/cache/ensemble_2026.npy, true_2026.npy, uncertainty_2026.npy ...
python backend/scripts/gate5_2026.py       # → backend/cache/metrics_2026.json (frozen 2025 quantiles)
```

### 7. Frontend

Build steps (need the data + ensemble cache from steps 2–6; on a fresh clone these
regenerate what the committed repo already ships):

```
python backend/scripts/prep_frontend_data.py --test2026
python backend/scripts/gen_metrics.py --test2026
python backend/scripts/write_html.py
```

Serve (works from the committed assets alone — no data rebuild required):

```
cd frontend
python -m http.server 8000          # open http://localhost:8000
```


## Acknowledgments

- NSIDC for the CDR v6 sea-ice concentration product (G02202 / NSIDC-0051)
- ECMWF / Copernicus Climate Change Service for ERA5 single-level reanalysis
- Copernicus Marine Service for GLORYS12V1 ocean reanalysis
- NCPOR for the problem statement and station context+
