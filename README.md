# Antarctic Uncertainty-Aware Maritime Route Optimization

> **SIH 2026 — Problem Statement 26059**  
> **Component**: Route Optimization / Maritime Route Planning  
> **Stage**: STEP 2D — Baseline Freeze (Frozen Deterministic Baseline)

---

## What This Project Is

This is the route optimization component of a larger Antarctic Maritime Navigation Decision Support System. The broader system will eventually combine:

- **Sea-ice concentration (SIC) forecasts** — developed by a teammate
- **Iceberg detection and trajectory prediction** — developed by a teammate
- **Ocean currents and weather data** — consumed from existing sources
- **Route optimization** — this project

The route optimizer must consume predictions from the SIC and iceberg models and compute safe, efficient routes for research vessels operating in Antarctic waters.

## What This Project Is NOT

- **NOT a basic A\* project.** A\* may later be used as a baseline or a component, but the final optimization method has NOT been selected yet.
- **NOT an ML training project.** We are not training neural networks in Step 1. The SIC and iceberg prediction models are separate teammate components.
- **NOT a visualization project.** The GUI/dashboard comes much later, if at all.

## Why Uncertainty Matters

SIC forecasts and iceberg trajectory predictions are inherently uncertain, especially at 24–48 hour horizons. A route that appears safe under the "expected" forecast may perform poorly if actual conditions diverge. This project investigates whether routing over **multiple plausible future scenarios** (rather than a single deterministic forecast) produces routes that are more robust to forecast errors.

This idea has precedent in general maritime weather routing (CVaR-based methods, ensemble forecasting) but has NOT been applied to Antarctic ice-aware routing in any known open-source implementation.

## Current Stage: STEP 2D — Frozen Deterministic Baseline

**Status: PASS — Baseline frozen.**

### Frozen Baseline Objective

For a route `P = [p0, p1, ..., pN]`, minimize:

```
env_cost(p0) + Σ(i=1..N) [env_cost(p_i) + w_distance * d(p_{i-1}, p_i)]
```

where:

```
env_cost(i) = w_sic * SIC(i)
            + w_iceberg * IcebergRisk(i)
            + w_wind * WindCost(i)
            + w_current * max(CurrentCost(i), 0)
```

**Current-cost handling:**
- Favorable current (`CurrentCost < 0`): zero penalty
- Neutral current (`CurrentCost = 0`): zero penalty
- Adverse current (`CurrentCost > 0`): proportional positive cost
- Implemented as `w_curr * np.maximum(current_cost, 0.0)` — no global shift

**Start-cell cost:** The start cell's environmental cost is included in the total objective (not omitted).

**Movement cost:** `w_distance * movement_distance`, where distance is 1.0 for cardinal and √2 for diagonal.

**Heuristic:** `h(n) = w_distance * EuclideanDistance(n, goal)` — consistent with edge cost definition.

**Baseline results (seed=42, 40×50 grid):**
- Start: (2, 2), Goal: (35, 45)
- Total cost: 96.1501
- Route length: 56.669 grid units
- Expanded nodes: 3138
- Waypoints: 44
- Weights: w_sic=1.0, w_ice=1.0, w_wind=1.0, w_curr=1.0, w_distance=1.0

**Synthetic-data limitation:** All results are on synthetic data, NOT real Antarctic data. This baseline exists for algorithm development and comparison.

**This is NOT the final robust optimizer.** It is a deterministic baseline for comparison.

### Previous Stages

- **STEP 2C:** Scenario-based route comparison — confirmed uncertainty materially affects route selection (20/20 routes differ, Jaccard overlap = 0.216)
- **STEP 2B:** Deterministic A* baseline implementation
- **STEP 2A:** Environmental grid representation and synthetic environment
- **STEP 1:** Research audit and project foundation

**What STEP 2A accomplishes:**
- `EnvironmentalGrid` class: a clean, layered 2D grid data structure
- Synthetic Antarctic-like environment generator (deterministic, reproducible)
- Validation tests for grid creation, dimensions, navigability, value ranges
- Visualization of navigability, SIC, iceberg risk, and uncertainty layers
- Ready to later accept real teammate model outputs (SIC predictions, iceberg trajectories)

### What is an EnvironmentalGrid?

An `EnvironmentalGrid` is a 2D spatial grid where each cell stores layered environmental information as NumPy arrays. It is the shared data structure that the route optimizer, risk model, and uncertainty modules will all read from.

**Required layers (always present):**
- `navigable` — boolean mask, True where a ship can traverse
- `lat`, `lon` — 1D coordinate arrays (centers of rows/columns)

**Optional layers (None if not available):**
- `sic_mean` — sea-ice concentration, 0 to 1 (deterministic estimate)
- `sic_uncertainty` — SIC standard deviation, >= 0 (uncertainty estimate)
- `iceberg_risk` — iceberg threat score, 0 to 1 (deterministic estimate)
- `iceberg_uncertainty` — iceberg positional uncertainty in km, >= 0
- `wind_cost` — wind-related traversal cost multiplier, >= 0
- `current_cost` — current-related traversal cost, negative = favorable

The representation is intentionally layer-optional. It does not require all layers to be present. This allows the grid to work with partial data and to gradually accept teammate model outputs as they become available.

### Why this design can later accept teammate outputs

The SIC prediction model produces mean SIC and uncertainty estimates. The iceberg model produces risk scores and positional uncertainty. These map directly to grid layers:
- SIC mean/uncertainty → `sic_mean` / `sic_uncertainty`
- Iceberg risk/uncertainty → `iceberg_risk` / `iceberg_uncertainty`

No format conversion is needed — teammate outputs are loaded into the same grid layers.

### Important: This is NOT calibrated real-world uncertainty

The synthetic uncertainty values are designed for algorithm development and testing. They do NOT represent calibrated real-world Antarctic forecast uncertainty. When real SIC and iceberg model outputs are available, the uncertainty values will come from those models.

## Previous Stage: STEP 1 — Research Audit + Project Foundation

**What STEP 1 accomplished:**
- Research audit of existing solutions (10 repos, 10+ papers)
- Identification of what is already common vs. potentially differentiated
- Project structure and development foundation
- No final algorithm selection yet

## Project Structure

```
antarctic_route_optimization/
├── main.py                  # Project verification (Step 1)
├── config.py                # Configuration placeholders
├── requirements.txt         # Minimal dependencies
├── research/                # Research audit and literature notes
├── data/                    # Environmental and test data
│   ├── raw/                 # Original downloaded data
│   ├── processed/           # Processed gridded data
│   └── test/                # Small test datasets
├── src/                     # Source code
│   ├── environment/         # Environmental grid construction
│   ├── routing/             # Routing algorithms
│   ├── risk/                # Risk calculation
│   ├── uncertainty/         # Uncertainty modeling
│   └── utils/               # Shared utilities
├── experiments/             # Experiment scripts
├── outputs/                 # Generated results
│   ├── routes/              # Saved route data
│   ├── metrics/             # Evaluation metrics
│   └── figures/             # Plots and visualizations
└── tests/                   # Test suite
```

## Configuration

All parameters in `config.py` are **placeholders** for Step 1. They will be refined after the research audit determines appropriate values.

## Quick Start

```bash
# Verify project environment
python main.py

# Run tests (87 tests)
python -m pytest tests/ -v

# Run deterministic baseline experiment
python experiments/test_baseline_astar.py

# Run scenario-based route comparison
python experiments/test_scenario_routing.py
```

## References

- See `research/existing_solutions.md` for the full research audit.
- Key academic references:
  - Nuñez et al. (2023) — CVaR for stochastic ship routing (IROS)
  - SINTEF (2025) — Arctic route planning under ice uncertainty with CVaR
  - Smith et al. (2025) — PolarRoute path planning (JAIR)
  - Ensemble-forecast uncertainty-robust framework (CIE, 2026)

## License

TBD
