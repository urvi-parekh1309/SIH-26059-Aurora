"""
Read-only API for the SIH2026059 IceRoute-Robust interactive demo (Flask).

Serves the REAL committed artifacts already in this repository:

    backend/cache/routing_sic_2026.npy     real SIC forecast (167, 173, 369)
    backend/cache/routing_metadata.json    routing grid / band definition
    backend/cache/dates_2026.npy           forecast timestamps
    outputs/final_demo/final_route.json    verified A* route on real SIC

No model is trained, no artifact is modified, and no synthetic data is ever
returned.  Routing reuses the repository's existing code
(``src.data.sic_forecast``, ``src.routing.astar``,
``src.routing.scenario_router``) rather than reimplementing the algorithm.

Endpoints
    GET /api/health
    GET /api/sic/metadata
    GET /api/sic/<timestep>[?format=b64|array]
    GET /api/route
    GET /api/route/at/<timestep>
    POST /api/route/optimize          {start_lat,start_lon,goal_lat,goal_lon,
                                      timestep[,snap,max_snap_cells]}
    POST /api/route/reroute           {original_route, new_timestep[,start,goal]}
    GET /api/route/profile/<timestep>
    GET /api/uncertainty/<timestep>[?horizon=0..2]
    GET /api/uncertainty/summary
    GET /api/models/ensemble
    GET /api/map/coastline
    GET /api/reroute/<timestep>[?origin_timestep=0]
    GET /api/limitations
    GET /                         built React app (frontend/route_demo/dist)

NaN encoding
------------
JSON has no NaN literal, so a raw ``NaN`` would silently become ``null`` (or
break strict parsers) and the browser could not tell "SIC = 0" from
"invalid cell".  Therefore each timestep is returned as:

    sic_u8  : base64 uint8, ``round(SIC * 255)``, NaN written as 0
    valid   : base64 packed bitmask, 1 = real measurement, 0 = INVALID
              (non-navigable land / ice shelf / outside the model domain)

``valid == 0`` is authoritative: such a cell is non-navigable and its
``sic_u8`` byte carries NO meaning.  It must never be rendered or routed as
open water.  ``?format=array`` returns the same field with explicit ``null``
for invalid cells, for inspection and testing.

Run:
    python -m backend.api.main              # http://127.0.0.1:8000
"""

from __future__ import annotations

import base64
import json
import math
import os
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS  # type: ignore

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import paths as data_paths                          # noqa: E402
from src.data.builder import build_grid_template           # noqa: E402
from src.data.layer_status import registry_to_dict        # noqa: E402
from src.data.sic_forecast import SICForecastField        # noqa: E402
from src.data.unified import build_route_grid             # noqa: E402
from src.routing.astar import astar                        # noqa: E402
from src.routing.cost import CostWeights                   # noqa: E402
from src.routing.scenario_router import (                  # noqa: E402
    jaccard_overlap, route_coverage,
)

CACHE = ROOT / "backend" / "cache"
ROUTE_JSON = ROOT / "outputs" / "final_demo" / "final_route.json"
AURORA_DIST = ROOT / "frontend" / "aurora" / "dist"
FRONTEND_DIST = AURORA_DIST

# ---------------------------------------------------------------------------
# Dataset location — resolved by configuration, never hardcoded here.
#
# The dataset lives at:  My Drive/SIH_26_Sanika/dataset
# which in Google Colab is  /content/drive/MyDrive/SIH_26_Sanika/dataset
#
# Resolution lives in src/data/paths.py so the loaders, the audit tooling and
# this API all agree. Point SIH_DATA_ROOT at any synced/mounted copy to use a
# local or cloud path instead:
#
#   SIH_DATA_ROOT=/content/drive/MyDrive/SIH_26_Sanika/dataset   (Colab)
#   SIH_DATA_ROOT=D:\SIH\dataset                                (mounted Drive)
#
# When the root cannot be resolved the API keeps serving the committed
# in-repository artifacts and reports every layer NOT_AVAILABLE. The browser
# never talks to Drive; only this backend does.
# ---------------------------------------------------------------------------
DATA_ROOT = data_paths.data_root() or ROOT
CMEMS_ROOT = os.environ.get("ARCTIC_CMEMS_ROOT") or os.environ.get(
    "SIH_CMEMS_ROOT") or None
#: Copernicus Ocean root for the date-aware loader, from configuration.
CMEMS_DATE_AWARE_ROOT = str(
    CMEMS_ROOT if CMEMS_ROOT is not None
    else (data_paths.layer_path("copernicus_ocean") or DATA_ROOT / "dataset" / "Copernicus_Ocean")
)
ICEBERG_CANDIDATES = [
    Path(os.environ["SIH_ICEBERG_PATH"]) if os.environ.get("SIH_ICEBERG_PATH")
    else None,
    data_paths.layer_path("icebergs"),
    DATA_ROOT / "dataset" / "iceberg",
    DATA_ROOT / "iceberg",
    ROOT / "backend" / "cache" / "iceberg_risk.npy",
]
#: Provenance of the dataset location, surfaced by /api/system/status.
def _resolved_root_str() -> Optional[str]:
    root = data_paths.data_root()
    return str(root) if root else None


DATASET_LOCATION = {
    "hint": data_paths.DATASET_LOCATION_HINT,
    "resolved_root": _resolved_root_str(),
    "resolution": data_paths.data_root_source(),
    "searched_roots": data_paths.root_candidates(),
    "env_var": data_paths.DATA_ROOT_ENVVAR,
}


ROUTE_START = datetime(2026, 1, 6, tzinfo=timezone.utc)
#: Row/col defaults are used ONLY by the GET endpoints, which take grid indices
#: directly and therefore cannot snap. The default start must therefore be a
#: cell that is navigable on the routing grid.
#:
#: Cell (164, 114) is Cape Town's own 0.25 deg cell, but the real GEBCO land
#: mask correctly marks it as land, so A* cannot start there; the geographic
#: endpoints in POST /api/route/optimize snap to the neighbouring ocean cell
#: (163, 114) and route fine. Pointing a row/col default at a land cell just
#: makes every GET route fail.
#:
#: (172, 368) is the start cell of the project's verified real-SIC baseline
#: route (outputs/final_demo/final_route.json) and is navigable, so it is the
#: default that matches the recorded verification.
DEFAULT_START = (172, 368)
DEFAULT_GOAL = (20, 82)      # Maitri offshore approach cell (routing_station_goals.json)

#: Cost weights are configurable so that no term is silently enabled.  A term
#: only reaches the route when its weight is above zero AND its layer carries
#: finite data; both facts are reported per request in ``layer_status``.
WEIGHT_ENVVARS = {
    "w_sic": "SIH_W_SIC",
    "w_ice": "SIH_W_ICE",
    "w_wind": "SIH_W_WIND",
    "w_curr": "SIH_W_CURR",
    "w_distance": "SIH_W_DISTANCE",
    "w_depth": "SIH_W_DEPTH",
    "w_unc": "SIH_W_UNC",
    "w_ice_class": "SIH_W_ICE_CLASS",
    "vessel_draft_m": "SIH_VESSEL_DRAFT_M",
}


def cost_weights() -> CostWeights:
    """Build :class:`CostWeights` from the environment, defaulting to the
    project's documented objective (``w_sic`` and ``w_distance`` only)."""
    values: Dict[str, float] = {}
    for name, envvar in WEIGHT_ENVVARS.items():
        raw = os.environ.get(envvar)
        if raw is None or raw == "":
            continue
        try:
            values[name] = float(raw)
        except ValueError:
            raise RouteRequestError(
                f"{envvar} must be a number, got {raw!r}", "invalid_weight",
                weight=name)
    return CostWeights(**values)

LIMITATIONS: Dict[str, str] = {
    "sic": "REAL committed forecast output (backend/cache/routing_sic_2026.npy). "
           "NaN cells are non-navigable and are never zero-filled.",
    "uncertainty": "REAL committed artifact (backend/cache/uncertainty_2026.npy, "
                   "3 lead-time horizons). Present on the grid; it reaches the "
                   "cost only when SIH_W_UNC > 0.",
    "depth": "NOT AVAILABLE in this deployment. The committed GEBCO land mask "
             "(routing_land_extension.npy) IS applied to the extension band, so "
             "land is not routable, but no GEBCO depth field is loaded, so no "
             "depth penalty is computed. Set SIH_DATA_ROOT and SIH_W_DEPTH>0 to "
             "activate it.",
    "wind": "NOT AVAILABLE. No ERA5/ECMWF dataset is reachable, so wind_cost is "
            "absent and contributes nothing. No wind is faked.",
    "cmems": "Unavailable in the current development environment. No currents "
             "are shown or faked. The CMEMS loader now also accepts 2026 "
             "forecast products (CMEMS_Future_Forecast), which no reanalysis "
             "covers.",
    "iceberg": "NOT AVAILABLE. src/data/adapters.py::IcebergAdapter reads a risk "
               "field, but no predictor output exists, so iceberg_risk is absent "
               "and the iceberg standoff term is not computed.",
    "ice_multiplier": "The committed POLARIS-style multiplier "
                      "(routing_multiplier_2026.npy) is applied only when "
                      "SIH_W_ICE_CLASS > 0. With the default 0 the SIC cost is "
                      "linear in SIC, so consolidated ice is not impassable.",
    "cvar": "UNAVAILABLE. src/uncertainty/cvar.py and "
            "src/routing/scenario_router.py::select_robust_route are valid code "
            "but are not reachable from this API, and "
            "src/uncertainty/scenarios.py::generate_scenarios requires "
            "iceberg_risk_uncertainty, which has no data source. No CVaR value "
            "is computed or claimed. At the default alpha=0.05 with 20 "
            "scenarios the tail size is 1, so CVaR would equal VaR equals the "
            "worst case rather than a tail mean.",
    "route_ml": "Policy checkpoint present (outputs/ml/route_policy.pt), "
                "trained on synthetic 20x25 smoke data; not used as a "
                "real-world accuracy claim and not used to produce this route.",
    "route": "Deterministic A* + CostMap on the real SIC field.",
    "land_mask": "The model band (-75..-50) carries a land mask through SIC NaN. "
                 "The extension band (-50..-32) is masked from the committed "
                 "GEBCO artifact. No GEBCO depth field is loaded.",
}



# ---------------------------------------------------------------------------
# Lazily-initialised real data (memory-mapped; nothing is rewritten)
# ---------------------------------------------------------------------------

_FIELD: Optional[SICForecastField] = None
_SIC_MMAP: Optional[np.ndarray] = None
_UNC_MMAP: Optional[np.ndarray] = None


def field() -> SICForecastField:
    global _FIELD
    if _FIELD is None:
        if not (CACHE / "routing_sic_2026.npy").exists():
            raise FileNotFoundError(
                "real SIC artifact backend/cache/routing_sic_2026.npy missing")
        _FIELD = SICForecastField(cache_dir=str(CACHE),
                                  route_start_datetime=ROUTE_START)
    return _FIELD


def sic_mmap() -> np.ndarray:
    global _SIC_MMAP
    if _SIC_MMAP is None:
        _SIC_MMAP = np.load(str(CACHE / "routing_sic_2026.npy"), mmap_mode="r")
    return _SIC_MMAP


def uncertainty_mmap() -> np.ndarray:
    """
    Memory-map the committed SIC forecast-uncertainty artifact.

    Shape (n_time, 3, 101, 361): 3 lead-time horizons on the ConvLSTM model
    band only (lat -75..-50, lon -10..80).  Never modified, never regenerated.
    """
    global _UNC_MMAP
    if _UNC_MMAP is None:
        _UNC_MMAP = np.load(str(CACHE / "uncertainty_2026.npy"), mmap_mode="r")
    return _UNC_MMAP


def native_grid():
    spec = field().grid_spec()
    return build_grid_template(lat_min=spec["lat_min"], lat_max=spec["lat_max"],
                               lon_min=spec["lon_min"], lon_max=spec["lon_max"],
                               resolution_deg=spec["resolution_deg"])


def load_route_artifact() -> Dict[str, Any]:
    if not ROUTE_JSON.exists():
        raise FileNotFoundError(
            f"verified route artifact {ROUTE_JSON.name} is not available")
    return json.loads(ROUTE_JSON.read_text(encoding="utf-8"))


def check_timestep(t: int) -> int:
    n = int(sic_mmap().shape[0])
    if t < 0 or t >= n:
        raise IndexError(
            f"timestep {t} out of range; the real artifact has {n} forecast "
            f"timesteps (0..{n - 1})")
    return t


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------

def encode_slice_b64(sic2d: np.ndarray) -> Dict[str, Any]:
    """uint8 value array + packed validity bitmask, both base64."""
    valid = ~np.isnan(sic2d)
    values = np.zeros(sic2d.shape, dtype=np.uint8)
    values[valid] = np.round(
        np.clip(sic2d[valid], 0.0, 1.0) * 255.0).astype(np.uint8)
    packed = np.packbits(valid.reshape(-1))  # row-major
    return {
        "shape": [int(sic2d.shape[0]), int(sic2d.shape[1])],
        "sic_u8": base64.b64encode(values.tobytes()).decode("ascii"),
        "valid": base64.b64encode(packed.tobytes()).decode("ascii"),
        "value_scale": 255,
    }


def encode_slice_array(sic2d: np.ndarray) -> List[List[Optional[float]]]:
    """Explicit ``null`` for every invalid cell (inspection / testing)."""
    return [[None if not np.isfinite(v) else round(float(v), 6) for v in row]
            for row in sic2d]


def slice_stats(sic2d: np.ndarray) -> Dict[str, Any]:
    finite = sic2d[np.isfinite(sic2d)]
    n_total = int(sic2d.size)
    return {
        "min": float(finite.min()) if finite.size else None,
        "mean": float(finite.mean()) if finite.size else None,
        "max": float(finite.max()) if finite.size else None,
        "n_cells": n_total,
        "n_navigable": int(finite.size),
        "n_non_navigable": int(n_total - finite.size),
    }


# ---------------------------------------------------------------------------
# Caching (per-timestep payloads and per-route plans)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=48)
def cached_slice_b64(t: int) -> Dict[str, Any]:
    return encode_slice_b64(np.asarray(sic_mmap()[t], dtype=np.float64))


@lru_cache(maxsize=48)
def cached_stats(t: int) -> Dict[str, Any]:
    return slice_stats(np.asarray(sic_mmap()[t], dtype=np.float64))


@lru_cache(maxsize=16)
def route_grid(t: int, weights_key: str):
    """
    Unified EnvironmentalGrid + CostMap + provenance for one real timestep.

    Every environmental layer that is genuinely reachable in this deployment is
    attached here and nowhere else, so the grid the optimizer consumes is the
    grid whose provenance is reported.  Layers whose dataset is missing stay
    absent and are reported ``NOT_AVAILABLE``.
    """
    weights = cost_weights()
    return build_route_grid(
        t, cache_dir=CACHE, route_start=ROUTE_START, weights=weights)


def _weights_key() -> str:
    return json.dumps(cost_weights().to_dict(), sort_keys=True)


@lru_cache(maxsize=16)
def plan_route(t: int, start_row: int, start_col: int,
               goal_row: int, goal_col: int, weights_key: str = "default") -> Dict[str, Any]:
    """Run the repository's existing A* + CostMap on one real SIC timestep."""
    grid, cost_map, report = route_grid(t, weights_key)
    result = astar(grid, (start_row, start_col), (goal_row, goal_col),
                   weights=cost_weights())
    path = [[int(r), int(c)] for r, c in result.path]
    stats: Dict[str, Any] = {"mean_sic": None, "max_sic": None,
                             "nan_cells_on_route": None}
    if path:
        rows = np.array([p[0] for p in path])
        cols = np.array([p[1] for p in path])
        route_sic = grid.sic_mean[rows, cols]
        stats = {
            "mean_sic": float(np.nanmean(route_sic)),
            "max_sic": float(np.nanmax(route_sic)),
            "nan_cells_on_route": int(np.isnan(route_sic).sum()),
        }
    return {
        "timestep": t,
        "success": bool(result.success),
        "waypoints": int(result.num_waypoints),
        "route_length_grid_units": float(result.route_length),
        "total_cost": float(result.total_cost),
        "expanded_nodes": int(result.expanded_nodes),
        "path": path,
        "layer_status": registry_to_dict(
            report["registry"],
            cost_layers=report["cost_breakdown"]["layers_in_cost"],
            extras={"land_mask": report["land_mask"]}),
        "cost_breakdown": report["cost_breakdown"],
        **stats,
    }


# ---------------------------------------------------------------------------
# Interactive route planning: geographic input + hard safety validation
#
# Every rule below is a REJECTION rule.  No NaN is ever zero-filled, no
# non-navigable cell is ever routed through, and nothing is silently snapped:
# when an endpoint sits on a non-navigable cell the caller is told exactly
# what happened (``snapped`` + ``snap_radius_cells``) or the request is
# refused with HTTP 400.
# ---------------------------------------------------------------------------

class RouteRequestError(ValueError):
    """Invalid or unsafe route request -> HTTP 400, never a fabricated route."""

    def __init__(self, message: str, reason: str, **detail: Any):
        super().__init__(message)
        self.reason = reason
        self.detail = detail

    def to_payload(self) -> Dict[str, Any]:
        return {"success": False, "error": str(self), "reason": self.reason,
                **self.detail}


#: How far (in cells, Chebyshev) an endpoint may be moved onto a navigable
#: cell when ``snap`` is requested.  3 cells = 0.75 deg, and the snap is
#: always reported back to the caller.
MAX_SNAP_CELLS = 3


def _nearest_navigable(grid, rc: Tuple[int, int],
                       max_radius: int) -> Tuple[Optional[Tuple[int, int]], int]:
    """Nearest navigable cell to ``rc`` within ``max_radius`` (8-connected)."""
    r0, c0 = rc
    for radius in range(0, max_radius + 1):
        best: Optional[Tuple[int, int, int]] = None
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if max(abs(dr), abs(dc)) != radius:
                    continue
                r, c = r0 + dr, c0 + dc
                if 0 <= r < grid.n_rows and 0 <= c < grid.n_cols \
                        and bool(grid.navigable[r, c]):
                    d2 = dr * dr + dc * dc
                    if best is None or d2 < best[0]:
                        best = (d2, r, c)
        if best is not None:
            return (best[1], best[2]), radius
    return None, -1


def resolve_endpoint(label: str, lat: Any, lon: Any, grid,
                     allow_snap: bool,
                     max_snap_cells: int) -> Dict[str, Any]:
    """
    Geographic coordinate -> validated (row, col) on the routing grid.

    Bounds validation is delegated to ``SICForecastField.latlon_to_rc``,
    which already refuses any coordinate with no cell centre within half a
    cell of the supported grid.
    """
    try:
        lat_f, lon_f = float(lat), float(lon)
    except (TypeError, ValueError):
        raise RouteRequestError(
            f"{label}: lat/lon must be numbers, got lat={lat!r} lon={lon!r}",
            "invalid_coordinate", endpoint=label)
    if not (math.isfinite(lat_f) and math.isfinite(lon_f)):
        raise RouteRequestError(
            f"{label}: lat/lon must be finite, got lat={lat_f} lon={lon_f}",
            "invalid_coordinate", endpoint=label)

    f = field()
    try:
        rc = f.latlon_to_rc(lat_f, lon_f)
    except IndexError as exc:
        raise RouteRequestError(str(exc), "out_of_grid", endpoint=label,
                                lat=lat_f, lon=lon_f)

    info: Dict[str, Any] = {
        "requested": {"lat": lat_f, "lon": lon_f},
        "cell": [int(rc[0]), int(rc[1])],
        "lat": float(f.rowcol_to_latlon(*rc)[0]),
        "lon": float(f.rowcol_to_latlon(*rc)[1]),
        "navigable": bool(grid.navigable[rc]),
        "snapped": False,
        "snap_radius_cells": 0,
    }

    if info["navigable"]:
        return info

    if not allow_snap:
        raise RouteRequestError(
            f"{label} cell {info['cell']} "
            f"({info['lat']:.2f}, {info['lon']:.2f}) is NON-NAVIGABLE "
            f"(land / ice shelf / no valid SIC) and snap=false",
            "endpoint_not_navigable", endpoint=label, **info)

    snapped, radius = _nearest_navigable(grid, rc, max_snap_cells)
    if snapped is None:
        raise RouteRequestError(
            f"{label} cell {info['cell']} "
            f"({info['lat']:.2f}, {info['lon']:.2f}) is NON-NAVIGABLE and no "
            f"navigable cell exists within {max_snap_cells} cells "
            f"({max_snap_cells * 0.25:.2f} deg)",
            "endpoint_not_navigable", endpoint=label, **info)

    info.update({
        "cell": [int(snapped[0]), int(snapped[1])],
        "lat": float(f.rowcol_to_latlon(*snapped)[0]),
        "lon": float(f.rowcol_to_latlon(*snapped)[1]),
        "navigable": True,
        "snapped": True,
        "snap_radius_cells": int(radius),
    })
    return info


def path_length_km(path: List[List[int]]) -> float:
    """Great-circle length of an 8-connected path on the real 0.25 deg grid."""
    if len(path) < 2:
        return 0.0
    lat = np.asarray(np.load(str(CACHE / "routing_lat.npy")), dtype=np.float64)
    lon = np.asarray(np.load(str(CACHE / "routing_lon.npy")), dtype=np.float64)
    total = 0.0
    for (r0, c0), (r1, c1) in zip(path, path[1:]):
        total += _gc_km(lat[r0], lon[c0], lat[r1], lon[c1])
    return total


def _gc_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    la0, lo0 = math.radians(lat1), math.radians(lon1)
    la1, lo1 = math.radians(lat2), math.radians(lon2)
    a = (math.sin((la1 - la0) / 2.0) ** 2
         + math.cos(la0) * math.cos(la1) * math.sin((lo1 - lo0) / 2.0) ** 2)
    return 2.0 * 6371.0 * math.asin(math.sqrt(min(1.0, a)))


def direct_length_km(start_rc: Tuple[int, int],
                     goal_rc: Tuple[int, int]) -> float:
    """Straight-line start-goal great-circle distance, for context."""
    lat = np.asarray(np.load(str(CACHE / "routing_lat.npy")), dtype=np.float64)
    lon = np.asarray(np.load(str(CACHE / "routing_lon.npy")), dtype=np.float64)
    return _gc_km(lat[start_rc[0]], lon[start_rc[1]],
                  lat[goal_rc[0]], lon[goal_rc[1]])


def validate_path(grid, path: List[List[int]], start_rc: Tuple[int, int],
                  goal_rc: Tuple[int, int],
                  cost_layers: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Post-route safety validation.  Returns the check report; raises
    ``RouteRequestError`` if any hard check fails, so a route is only ever
    returned after it has been proven in-bounds, connected, goal-reaching
    and free of NaN / non-navigable cells.
    """
    n_rows, n_cols = grid.n_rows, grid.n_cols
    report: Dict[str, Any] = {}

    if not path:
        raise RouteRequestError(
            "no route exists between the requested start and goal on this "
            "SIC field (all connecting cells are non-navigable)",
            "no_route", start=list(start_rc), goal=list(goal_rc))

    rows = np.array([p[0] for p in path], dtype=np.int64)
    cols = np.array([p[1] for p in path], dtype=np.int64)

    in_bounds = bool(((rows >= 0) & (rows < n_rows)
                      & (cols >= 0) & (cols < n_cols)).all())
    report["all_cells_in_bounds"] = in_bounds
    if not in_bounds:
        raise RouteRequestError("route leaves the routing grid", "out_of_grid")

    report["starts_at_start"] = (int(path[0][0]), int(path[0][1])) == tuple(start_rc)
    report["reaches_goal"] = (int(path[-1][0]), int(path[-1][1])) == tuple(goal_rc)
    if not report["reaches_goal"]:
        raise RouteRequestError(
            f"route does not terminate at the goal cell {list(goal_rc)}",
            "goal_not_reached", path_end=[int(path[-1][0]), int(path[-1][1])])

    max_step = 0
    for (r0, c0), (r1, c1) in zip(path, path[1:]):
        max_step = max(max_step, abs(r1 - r0), abs(c1 - c0))
    report["max_step_cells"] = int(max_step)
    report["contiguous_8_connected"] = bool(max_step <= 1)
    if max_step > 1:
        raise RouteRequestError(
            f"route is not 8-connected (largest jump {max_step} cells)",
            "disconnected_path")

    non_navigable = int((~np.asarray(grid.navigable)[rows, cols]).sum())
    route_sic = np.asarray(grid.sic_mean, dtype=np.float64)[rows, cols]
    nan_cells = int(np.isnan(route_sic).sum())
    report["non_navigable_cells"] = non_navigable
    report["nan_cells"] = nan_cells
    if non_navigable or nan_cells:
        raise RouteRequestError(
            f"route contains {non_navigable} non-navigable and {nan_cells} NaN "
            f"cells; refusing to return it (NaN is never zero-filled)",
            "unsafe_route_cells",
            non_navigable_cells=non_navigable, nan_cells=nan_cells)

    # SIC safety is the hard requirement and is enforced above: no NaN SIC, no
    # non-navigable cell, contiguous 8-connected geometry that reaches the goal.
    #
    # Everything else is an OPTIONAL cost term. A layer may therefore be real
    # but only partially covering the grid (uncertainty_2026.npy is 101x361 on
    # a 173x369 routing grid, so it does not reach the extension band). That
    # must not veto a route — but it must also never be hidden:
    #
    #   w_unc == 0  -> the term did not reach the cost; the route stands and
    #                  the uncovered cells are reported.
    #   w_unc  > 0  -> the term IS required where it is applied; if coverage is
    #                  insufficient the request fails explicitly rather than
    #                  quietly dropping the penalty.
    cost_layers = set(cost_layers or ())
    layer_coverage: Dict[str, Any] = {}
    insufficient: Dict[str, int] = {}
    for name in grid.LAYER_NAMES:
        arr = getattr(grid, name, None)
        if arr is None:
            continue
        values = np.asarray(arr, dtype=np.float64)[rows, cols]
        missing = int((~np.isfinite(values)).sum())
        layer_coverage[name] = {
            "covered_route_cells": int(len(path) - missing),
            "uncovered_route_cells": missing,
            "in_cost": name in cost_layers,
        }
        if missing and name in cost_layers:
            insufficient[name] = missing

    report["layer_coverage"] = layer_coverage
    report["layers_in_cost"] = sorted(cost_layers)
    if insufficient:
        detail = ", ".join(f"{k} ({v})" for k, v in sorted(insufficient.items()))
        raise RouteRequestError(
            "an uncertainty-aware route was requested, but the enabled cost "
            f"layer(s) {detail} have no finite value on those route cells. "
            "The route is refused rather than priced with a missing term; no "
            "value is ever fabricated to fill the gap.",
            "insufficient_enabled_layer_coverage",
            layers=insufficient, enabled_weights=cost_weights().to_dict(),
            layer_coverage=layer_coverage)

    return report


def changed_segments(orig_path: List[List[int]],
                     new_path: List[List[int]]) -> List[Dict[str, Any]]:
    """
    Contiguous runs of cells that one route uses and the other does not.

    Reported per route, in that route's own path order, so the frontend can
    draw exactly which stretch of track was abandoned and which stretch was
    newly taken. Nothing is interpolated: a segment is always a real run of
    real cells from one of the two returned paths.
    """
    orig_cells = {tuple(p) for p in orig_path}
    new_cells = {tuple(p) for p in new_path}

    def runs(path, other):
        out: List[Dict[str, Any]] = []
        run: List[List[int]] = []
        for cell in path:
            if tuple(cell) in other:
                if run:
                    out.append(run)
                    run = []
            else:
                run.append(cell)
        if run:
            out.append(run)
        return out

    segments: List[Dict[str, Any]] = []
    for label, path, other in (("original", orig_path, new_cells),
                               ("updated", new_path, orig_cells)):
        for run_cells in runs(path, other):
            first, last = run_cells[0], run_cells[-1]
            length = 0.0
            for (r0, c0), (r1, c1) in zip(run_cells, run_cells[1:]):
                length += math.hypot(r1 - r0, c1 - c0)
            segments.append({
                "route": label,
                "cells": len(run_cells),
                "from_cell": [int(first[0]), int(first[1])],
                "to_cell": [int(last[0]), int(last[1])],
                "length_grid_units": round(length, 3),
            })
    return segments


def optimize_route(start_lat: Any, start_lon: Any, goal_lat: Any, goal_lon: Any,
                   timestep: Any, allow_snap: bool = True,
                   max_snap_cells: int = MAX_SNAP_CELLS) -> Dict[str, Any]:
    """
    Full interactive path: geographic request -> validated environmental grid
    -> A* + CostMap on the REAL SIC field -> validated route.

    Shares ``plan_route`` (and therefore its cache) with the GET endpoints,
    so there is exactly one routing implementation in the project.
    """
    f = field()
    try:
        t = int(timestep)
    except (TypeError, ValueError):
        raise RouteRequestError(
            f"timestep must be an integer, got {timestep!r}",
            "invalid_timestep", timestep=timestep)
    try:
        check_timestep(t)
    except IndexError as exc:
        raise RouteRequestError(str(exc), "timestep_out_of_range", timestep=t)

    grid, cost_map, report = route_grid(t, _weights_key())

    start = resolve_endpoint("start", start_lat, start_lon, grid,
                             allow_snap, max_snap_cells)
    goal = resolve_endpoint("goal", goal_lat, goal_lon, grid,
                            allow_snap, max_snap_cells)
    start_rc = (start["cell"][0], start["cell"][1])
    goal_rc = (goal["cell"][0], goal["cell"][1])

    wk = _weights_key()
    plan = plan_route(t, start_rc[0], start_rc[1], goal_rc[0], goal_rc[1], wk)
    if not plan["success"]:
        raise RouteRequestError(
            "A* found no route between the requested endpoints on this SIC "
            "field", "no_route", start=list(start_rc), goal=list(goal_rc))

    path: List[List[int]] = plan["path"]
    validation = validate_path(grid, path, start_rc, goal_rc,
                               cost_layers=plan.get("cost_breakdown", {})
                               .get("layers_in_cost"))

    spec = f.grid_spec()
    return {
        "success": True,
        "timestep": t,
        "date": str(f.dates[t])[:10],
        "start": {"lat": start["lat"], "lon": start["lon"],
                  "cell": start["cell"], "snapped": start["snapped"],
                  "snap_radius_cells": start["snap_radius_cells"],
                  "requested": start["requested"]},
        "goal": {"lat": goal["lat"], "lon": goal["lon"],
                 "cell": goal["cell"], "snapped": goal["snapped"],
                 "snap_radius_cells": goal["snap_radius_cells"],
                 "requested": goal["requested"]},
        "path": path,
        "waypoints": int(plan["waypoints"]),
        "route_length": float(plan["route_length_grid_units"]),
        "route_length_units": "grid_cells",
        "route_length_km": round(path_length_km(path), 2),
        "route_length_km_note": "great-circle distance summed along the "
                                "8-connected staircase path; always >= the "
                                "straight-line start-goal distance",
        "direct_length_km": round(direct_length_km(start_rc, goal_rc), 2),
        "mean_sic": plan["mean_sic"],
        "max_sic": plan["max_sic"],
        "nan_cells": validation["nan_cells"],
        "non_navigable_cells": validation["non_navigable_cells"],
        "total_cost": plan["total_cost"],
        "expanded_nodes": plan["expanded_nodes"],
        "validation": validation,
        "algorithm": "A* + CostMap (src/routing/astar.py, src/routing/cost.py)",
        "cost_weights": cost_weights().to_dict(),
        "cost_breakdown": report["cost_breakdown"],
        "layer_status": plan["layer_status"],
        "environmental_grid": report["grid"],
        "grid": {"n_rows": spec["n_rows"], "n_cols": spec["n_cols"],
                 "resolution_deg": spec["resolution_deg"]},
        "provenance": {
            "sic_field": "REAL committed 2026 SIC forecast output "
                         "(backend/cache/routing_sic_2026.npy)",
            "sic_model": "3-seed ConvLSTM ensemble (backend/runs/"
                         "final_10ch_3f_seed{0,1,2}/best_model.pt); "
                         "inference is NOT re-run at request time because the "
                         "raw 2026 multi-channel inputs are absent",
            "route": "A* + CostMap on the real SIC field",
            "nan_policy": "NaN = non-navigable; never zero-filled, never routed",
            "current_policy": "currents are NOT part of this cost "
                              "(CMEMS unavailable); no current is faked",
            "grid_assembly": "src/data/unified.py::build_route_grid",
            "land_mask": report["land_mask"],
        },
        "limitations": LIMITATIONS,
    }


def reroute_route(original: Any, new_timestep: Any,
                  start: Optional[Any] = None, goal: Optional[Any] = None,
                  allow_snap: bool = True) -> Dict[str, Any]:
    """
    Re-optimize on a LATER real SIC field and report what actually changed.

    The endpoints are taken from ``original`` (or overridden by ``start`` /
    ``goal``), so the comparison is like-for-like: same leg, new environment.
    Both plans come from the same ``plan_route`` A* + CostMap call, and both
    are safety-validated before anything is returned.
    """
    f = field()
    if not isinstance(original, dict):
        raise RouteRequestError(
            "original_route must be an object with at least a 'path' array",
            "invalid_original_route")

    orig_path = original.get("path")
    if not orig_path or not isinstance(orig_path, list):
        raise RouteRequestError(
            "original_route.path is required (use the path returned by "
            "POST /api/route/optimize)", "invalid_original_route")

    try:
        nt = int(new_timestep)
        check_timestep(nt)
    except (TypeError, ValueError):
        raise RouteRequestError(
            f"new_timestep must be an integer, got {new_timestep!r}",
            "invalid_timestep", new_timestep=new_timestep)
    except IndexError as exc:
        raise RouteRequestError(str(exc), "timestep_out_of_range",
                                new_timestep=nt)

    origin_step = original.get("timestep")
    origin_step = int(origin_step) if origin_step is not None else 0
    try:
        check_timestep(origin_step)
    except IndexError:
        origin_step = 0

    if nt < origin_step:
        raise RouteRequestError(
            f"new_timestep {nt} is earlier than the original route's timestep "
            f"{origin_step}; reroute must move forward in the forecast",
            "timestep_not_forward", original_timestep=origin_step,
            new_timestep=nt)

    # Endpoints: explicit override > original_route endpoints > its path ends.
    def endpoint(src, fallback_path_index):
        if isinstance(src, dict) and "lat" in src and "lon" in src:
            return float(src["lat"]), float(src["lon"])
        cell = src.get("cell") if isinstance(src, dict) else None
        if cell is None:
            cell = orig_path[fallback_path_index]
        return f.rowcol_to_latlon(int(cell[0]), int(cell[1]))

    start_latlon = endpoint(start if start is not None else original.get("start"), 0)
    goal_latlon = endpoint(goal if goal is not None else original.get("goal"), -1)

    wk = _weights_key()
    grid, cost_map, report = route_grid(nt, wk)
    s_info = resolve_endpoint("start", start_latlon[0], start_latlon[1], grid,
                              allow_snap, MAX_SNAP_CELLS)
    g_info = resolve_endpoint("goal", goal_latlon[0], goal_latlon[1], grid,
                              allow_snap, MAX_SNAP_CELLS)
    src = (s_info["cell"][0], s_info["cell"][1])
    dst = (g_info["cell"][0], g_info["cell"][1])

    new_plan = plan_route(nt, src[0], src[1], dst[0], dst[1], wk)
    if not new_plan["success"]:
        raise RouteRequestError(
            f"no route exists between the same endpoints on the D{nt} SIC "
            f"field", "no_route", new_timestep=nt)

    new_path: List[List[int]] = new_plan["path"]
    new_validation = validate_path(
        grid, new_path, src, dst,
        cost_layers=new_plan.get("cost_breakdown", {}).get("layers_in_cost"))

    orig_cells = [[int(r), int(c)] for r, c in orig_path]
    both = bool(orig_cells and new_path)

    def block(plan_payload, path, validation, s_i, g_i, step):
        return {
            "timestep": step,
            "date": str(f.dates[step])[:10] if 0 <= step < len(f.dates) else None,
            "success": bool(plan_payload.get("success", True)),
            "waypoints": len(path),
            "route_length": plan_payload.get("route_length_grid_units"),
            "route_length_units": "grid_cells",
            "route_length_km": round(path_length_km(path), 2),
            "direct_length_km": round(direct_length_km(src, dst), 2),
            "mean_sic": plan_payload.get("mean_sic"),
            "max_sic": plan_payload.get("max_sic"),
            "nan_cells": validation.get("nan_cells") if validation else None,
            "non_navigable_cells": validation.get("non_navigable_cells")
            if validation else None,
            "total_cost": plan_payload.get("total_cost"),
            "expanded_nodes": plan_payload.get("expanded_nodes"),
            "start": s_i,
            "goal": g_i,
            "path": path,
            "validation": validation,
        }

    # The original route is supplied by the client, so it is validated here
    # against the grid of its own timestep before it is echoed back as a
    # "route". An unvalidated path must never be presented as one.
    origin_grid = f.load(t_hours=float(origin_step) * 24.0,
                         grid_template=native_grid())
    origin_validation = validate_path(
        origin_grid, orig_cells,
        (orig_cells[0][0], orig_cells[0][1]),
        (orig_cells[-1][0], orig_cells[-1][1]),
        cost_layers=plan_route(
            origin_step, orig_cells[0][0], orig_cells[0][1],
            orig_cells[-1][0], orig_cells[-1][1], wk
        ).get("cost_breakdown", {}).get("layers_in_cost"))
    origin_sic = np.asarray(origin_grid.sic_mean, dtype=np.float64)[
        [p[0] for p in orig_cells], [p[1] for p in orig_cells]]

    original_block = {
        "timestep": origin_step,
        "date": str(f.dates[origin_step])[:10],
        "waypoints": len(orig_cells),
        "path": orig_cells,
        "mean_sic": float(np.nanmean(origin_sic)),
        "max_sic": float(np.nanmax(origin_sic)),
        "nan_cells": origin_validation["nan_cells"],
        "non_navigable_cells": origin_validation["non_navigable_cells"],
        "validation": origin_validation,
        "source": "the client's plan, re-validated against the D%d SIC field"
                  % origin_step,
    }
    updated_block = block(new_plan, new_path, new_validation, s_info, g_info, nt)

    comparison = {
        "identical_path": both and orig_cells == new_path,
        "jaccard_overlap": float(jaccard_overlap([tuple(p) for p in orig_cells],
                                                 [tuple(p) for p in new_path]))
        if both else None,
        "route_coverage": float(route_coverage([tuple(p) for p in orig_cells],
                                               [tuple(p) for p in new_path]))
        if both else None,
        "changed_cells": len({tuple(p) for p in orig_cells}
                             ^ {tuple(p) for p in new_path}) if both else None,
        "waypoints_before": len(orig_cells),
        "waypoints_after": len(new_path),
    }

    return {
        "success": True,
        "status": "SUCCESS",
        "origin_timestep": origin_step,
        "new_timestep": nt,
        "forecast_step_days": nt - origin_step,
        "origin_date": original_block["date"],
        "new_date": updated_block["date"],
        "start": s_info,
        "goal": g_info,
        "original_route": original_block,
        "updated_route": updated_block,
        "route_comparison": comparison,
        "changed_segments": changed_segments(orig_cells, new_path),
        "layer_status": new_plan["layer_status"],
        "cost_breakdown": report["cost_breakdown"],
        "environmental_grid": report["grid"],
        "cost_weights": cost_weights().to_dict(),
        "land_mask": report["land_mask"],
        "metrics": {
            "original": {k: original_block.get(k) for k in
                         ("waypoints", "mean_sic", "max_sic", "route_length")},
            "updated": {k: updated_block.get(k) for k in
                        ("waypoints", "mean_sic", "max_sic", "route_length",
                         "total_cost")},
            "delta": {
                "mean_sic": (None if original_block.get("mean_sic") is None
                             or new_plan["mean_sic"] is None
                             else new_plan["mean_sic"] - original_block["mean_sic"]),
                "max_sic": (None if original_block.get("max_sic") is None
                            or new_plan["max_sic"] is None
                            else new_plan["max_sic"] - original_block["max_sic"]),
                "length_grid_units": (
                    None if original_block.get("route_length") is None
                    else new_plan["route_length_grid_units"]
                    - original_block["route_length"]),
            },
        },
        "algorithm": "A* + CostMap re-planned on the new real SIC field",
        "environment_change_note":
            "The route is recomputed against a different real SIC forecast "
            "timestep. An unchanged corridor is a real result, not a cached "
            "one: it means the cost surface did not move the optimum.",
        "limitations": LIMITATIONS,
    }


COASTLINE_JSON = ROOT / "frontend" / "data" / "coastline.json"


@lru_cache(maxsize=1)
def coastline() -> Dict[str, Any]:
    """
    Antarctic coastline context for the chart (read-only GeoJSON).

    Served from the repository's own Natural Earth coastline file so the map
    has real geographic context — the continent and its islands — instead of
    an empty black plate. It is pure cartography: no SIC, no routing, and it
    is never used by the optimizer.
    """
    if not COASTLINE_JSON.is_file():
        raise FileNotFoundError("coastline.json is not available")
    payload = json.loads(COASTLINE_JSON.read_text(encoding="utf-8"))
    payload["source"] = "frontend/data/coastline.json (Natural Earth coastline)"
    payload["role"] = ("map context only — not an input to the cost map or "
                       "the route")
    return payload


# ---------------------------------------------------------------------------
# Availability discovery (never optimistic, never fabricated)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def discover_cmems() -> Tuple[bool, str, Tuple[str, ...]]:
    """
    Look for real CMEMS GLORYS files under the configured data root.

    Returns (available, description, sample_files).  Uses the same layout
    the existing loader expects, so nothing is re-implemented.
    """
    root = Path(CMEMS_DATE_AWARE_ROOT)
    if not root.is_dir():
        return False, f"{root} (not mounted / not configured)", ()
    found: List[str] = []
    for pattern in ("*/*.nc", "**/*.nc"):
        for p in sorted(root.glob(pattern)):
            found.append(str(p))
            if len(found) >= 8:
                break
        if found:
            break
    if not found:
        return False, f"{root} (present but contains no .nc files)", ()
    return True, str(root), tuple(found)


@lru_cache(maxsize=1)
def discover_sic_checkpoints() -> Tuple[Dict[str, Any], ...]:
    """
    Locate independently trained SIC forecasting checkpoints (read-only).

    ``DATA_ROOT/models`` is a shared tree, so it also holds checkpoints that
    belong to other components (the YOLOv8 iceberg detector under
    ``models/iceberg/``).  Each ``.pt`` is probed for the ConvLSTM SIC
    signature and anything that is not a SIC forecaster is left out, so this
    list only ever contains SIC forecasting checkpoints.
    """
    out: List[Dict[str, Any]] = []
    search_roots = [ROOT / "backend" / "runs", DATA_ROOT / "models",
                    DATA_ROOT / "models" / "RL_final"]
    seen: set = set()
    for base in search_roots:
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*.pt")):
            rp = p.resolve()
            if rp in seen:
                continue
            seen.add(rp)
            entry: Dict[str, Any] = {
                "path": str(p.relative_to(ROOT)) if str(p).startswith(str(ROOT))
                        else str(p),
                "is_sic_forecaster": False,
            }
            try:  # inspect the state dict without executing anything
                import torch
                sd = torch.load(str(p), map_location="cpu", weights_only=True)
                keys = list(sd.keys()) if isinstance(sd, dict) else []
                entry["is_sic_forecaster"] = (
                    any(k.startswith("cells.") for k in keys)
                    and "head.weight" in keys
                )
                entry["n_tensors"] = len(keys)
            except Exception as exc:
                entry["error"] = f"{type(exc).__name__}"
            if entry["is_sic_forecaster"]:
                out.append(entry)
    return tuple(out)


@lru_cache(maxsize=1)
def discover_iceberg() -> Dict[str, Any]:
    """Look for iceberg risk / uncertainty layers needed by CVaR."""
    for cand in ICEBERG_CANDIDATES:
        if cand is not None and Path(cand).exists():
            return {
                "available": True,
                "path": str(cand),
                "label": "Iceberg risk data available",
            }
    return {
        "available": False,
        "path": None,
        "label": "Iceberg risk data NOT available",
        "searched": [str(c) for c in ICEBERG_CANDIDATES if c is not None],
        "note": ("src/data/adapters.py::IcebergAdapter is an empty stub and "
                 "routing_metadata.json marks the iceberg standoff cost term "
                 "as deferred."),
    }


@lru_cache(maxsize=1)
def discover_route_policy() -> Dict[str, Any]:
    """
    Describe the route ML policy checkpoint (never as a real-accuracy claim).
    """
    path = ROOT / "outputs" / "ml" / "route_policy.pt"
    if not path.exists():
        # *.pt is gitignored and the checkpoint was never committed, so the
        # honest answer is "missing" - with the same safety claims the loaded
        # branch makes, because neither case ever drives the shipped route.
        return {
            "present": False,
            "label": "Route ML policy: MISSING",
            "is_real_antarctic_accuracy": False,
            "used_for_final_route": False,
            "note": "outputs/ml/route_policy.pt is gitignored (*.pt) and was "
                    "never committed; no accuracy figure is claimed.",
        }
    info: Dict[str, Any] = {
        "present": True,
        "path": "outputs/ml/route_policy.pt",
        "architecture": "Linear(16->64) ReLU Dropout(0.15) "
                        "Linear(64->32) ReLU Linear(32->8)",
        "n_features": None,
        "n_actions": None,
        "training_data": "synthetic 20x25 smoke-test dataset "
                         "(outputs/ml/route_training_dataset.npz)",
        "is_real_antarctic_accuracy": False,
        "label": "Route ML policy: available (trained on synthetic smoke data)",
        "used_for_final_route": False,
        "note": "The final route is produced by A* + CostMap on real SIC. The "
                "policy is safety-constrained by the expert route; its score "
                "is not presented as real-world accuracy.",
    }
    try:
        import torch
        ck = torch.load(str(path), map_location="cpu", weights_only=False)
        info["n_features"] = int(ck.get("n_features", 0)) or None
        info["n_actions"] = int(ck.get("n_actions", 0)) or None
        info["training_val_accuracy"] = float(ck.get("best_val_accuracy", 0.0))
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


# ---------------------------------------------------------------------------
# Forecast-uncertainty, ensemble and route-profile helpers (read-only)
# ---------------------------------------------------------------------------

def uncertainty_frame(timestep: int, horizon: int) -> np.ndarray:
    """
    Expand one ``uncertainty_2026.npy`` slice to the full routing grid.

    The artifact is (n_time, 3, 101, 361): the ConvLSTM model band only.
    Rows 101..172 (lat -49.75..-32) and columns 361..368 (lon 80.25..82) were
    never inside the model domain, so they are returned as NaN — exactly the
    same convention ``SICForecastField.uncertainty_at_time`` uses, and the same
    "NaN means no value" contract the SIC endpoint already follows.
    """
    unc = uncertainty_mmap()
    n_time, n_h, n_mlat, n_mlon = (int(v) for v in unc.shape)
    if not 0 <= horizon < n_h:
        raise IndexError(
            f"horizon {horizon} out of range; the artifact has {n_h} horizons "
            f"(0..{n_h - 1})")
    check_timestep(timestep)
    n_rows, n_cols = (int(v) for v in sic_mmap().shape[1:])
    out = np.full((n_rows, n_cols), np.nan, dtype=np.float64)
    out[0:n_mlat, 0:n_mlon] = np.asarray(unc[timestep, horizon], dtype=np.float64)
    return out


def uncertainty_stats(unc2d: np.ndarray) -> Dict[str, Any]:
    """Descriptive statistics of one uncertainty frame (finite cells only)."""
    finite = unc2d[np.isfinite(unc2d)]
    n_total = int(unc2d.size)
    out: Dict[str, Any] = {
        "n_cells": n_total,
        "n_within_model_domain": int(finite.size),
        "n_outside_model_domain": int(n_total - finite.size),
        "min": None, "mean": None, "median": None, "max": None,
        "p90": None, "p99": None,
    }
    if finite.size:
        out.update({
            "min": float(finite.min()),
            "mean": float(finite.mean()),
            "median": float(np.median(finite)),
            "max": float(finite.max()),
            "p90": float(np.percentile(finite, 90)),
            "p99": float(np.percentile(finite, 99)),
        })
    return out


@lru_cache(maxsize=1)
def discover_ensemble() -> Dict[str, Any]:
    """
    Read the three independently trained SIC ConvLSTM checkpoints' own
    training metrics (read-only JSON written by the training run).

    Nothing is re-run here and no accuracy figure is recomputed; the numbers
    are exactly what ``backend/runs/*/metrics.json`` recorded.
    """
    runs_dir = ROOT / "backend" / "runs"
    members: List[Dict[str, Any]] = []
    for d in sorted(runs_dir.glob("final_10ch_3f_seed*")):
        metrics_path = d / "metrics.json"
        ckpt_path = d / "best_model.pt"
        entry: Dict[str, Any] = {
            "run": d.name,
            "seed": int(d.name.rsplit("seed", 1)[-1]) if "seed" in d.name else None,
            "checkpoint": str(ckpt_path.relative_to(ROOT)) if ckpt_path.exists() else None,
            "checkpoint_present": ckpt_path.is_file(),
        }
        if metrics_path.is_file():
            try:
                entry["metrics"] = json.loads(metrics_path.read_text(encoding="utf-8"))
            except Exception as exc:
                entry["metrics_error"] = f"{type(exc).__name__}"
        members.append(entry)
    verification: Dict[str, Any] = {}
    val_path = ROOT / "outputs" / "final_demo" / "final_system_validation.json"
    if val_path.is_file():
        try:
            val = json.loads(val_path.read_text(encoding="utf-8"))
            verification = {
                "source": "outputs/final_demo/final_system_validation.json",
                "tests_passed": val.get("tests_passed"),
                "tests_failed": val.get("tests_failed"),
                "tests_status": val.get("tests_status"),
                "overall_status": val.get("overall_status"),
                "warnings": val.get("warnings"),
                "cvar_computed": val.get("cvar_computed"),
            }
        except Exception as exc:
            verification = {"error": f"{type(exc).__name__}"}
    return {
        "available": bool(members),
        "n_members": len(members),
        "members": members,
        "verification": verification,
        "architecture": "ConvLSTMCell(10ch x3frame -> 32) k3p1 -> "
                        "ConvLSTMCell(32 -> 64) k3p1 -> Dropout2d(0.1) -> "
                        "Conv2d(64 -> 3, k1) + sigmoid",
        "param_count": 270147,
        "param_count_source": "outputs/final_demo/final_system_validation.json",
        "horizons": 3,
        "inference_rerun_possible": False,
        "note": "The committed 2026 forecast artifact (routing_sic_2026.npy / "
                "uncertainty_2026.npy) is what the routing pipeline consumes. "
                "The raw 2026 multi-channel inference inputs (SIC, ERA5, CMEMS) "
                "are not present in this deployment, so ConvLSTM inference is "
                "NOT re-run by this API. These are the checkpoints' own recorded "
                "training metrics.",
    }


@lru_cache(maxsize=1)
def discover_uncertainty_summary() -> Dict[str, Any]:
    """The committed uncertainty distribution summary (read-only JSON)."""
    p = CACHE / "uncertainty_stats.json"
    if not p.is_file():
        return {"available": False}
    d = json.loads(p.read_text(encoding="utf-8"))
    hist = d.get("histogram") or {}
    return {
        "available": True,
        "source": "backend/cache/uncertainty_stats.json",
        "miz_mean_std": d.get("miz_mean_std"),
        "miz_median_std": d.get("miz_median_std"),
        "miz_p10_std": d.get("miz_p10_std"),
        "miz_p90_std": d.get("miz_p90_std"),
        "full_mean_std": d.get("full_mean_std"),
        "histogram_bins": hist.get("bins"),
        "histogram_counts": hist.get("counts"),
        "units": "SIC fraction (dimensionless, 0-1); the artifact provides no "
                 "physical unit beyond the SIC scale",
    }


#: SIC band edges used by the project's own POLARIS-style ice classifier
#: (backend/scripts/build_routing_grid.py:168-175).  Exposed so the UI never
#: has to invent a threshold.
SIC_BANDS: List[Tuple[str, float, float]] = [
    ("open_water", 0.00, 0.15),
    ("marginal_ice", 0.15, 0.40),
    ("moderate_pack", 0.40, 0.70),
    ("hard_pack", 0.70, 0.85),
    ("impassable", 0.85, 1.01),
]
#: SIC value at/above which a cell counts as "high ice" for the route report.
SIC_HIGH_THRESHOLD = 0.40


def _jsonable(x: Any) -> Any:
    """
    Serialisable number: finite floats survive, NaN/Inf become ``None``.

    ``jsonify`` would otherwise emit the non-standard tokens ``NaN``/``Infinity``
    which the browser's ``JSON.parse`` rejects.
    """
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _sic_band(value: Optional[float]) -> str:
    """Classify one SIC sample with the repository's committed band edges."""
    if value is None:
        return "invalid"
    for name, lo, hi in SIC_BANDS:
        if lo <= value < hi:
            return name
    return "impassable"


@lru_cache(maxsize=8)
def route_profile(t: int, start_row: int, start_col: int,
                  goal_row: int, goal_col: int) -> Dict[str, Any]:
    """
    SIC encountered *along* the real A* route, sample by sample.

    Each sample carries the real grid cell, its real SIC at that timestep and
    the cumulative geodesic distance from the start, computed with the same
    great-circle cell sizes the repository builds (``routing_dx`` /
    ``routing_dy``, i.e. 0.25 deg at the cell's own latitude).  No value here
    is synthesised — if a cell is NaN the sample is reported as null and is
    never filled in.

    The response also carries the statistics a SIC report needs: min/mean/max,
    the distance share of the route in each committed ice band, the count and
    share of high-ice cells (``sic >= 0.40``, the moderate-pack onset in the
    project's own classifier) and the route's objective split between the SIC
    term, the distance term and everything else.
    """
    grid, cost_map, _report = route_grid(t, _weights_key())
    weights = cost_weights()
    result = astar(grid, (start_row, start_col), (goal_row, goal_col),
                   weights=weights)
    path = [(int(r), int(c)) for r, c in result.path]
    lat = np.asarray(grid.lat)
    lon = np.asarray(grid.lon)
    R_KM = 6371.0
    samples: List[Dict[str, Any]] = []
    cum = 0.0
    max_idx = -1
    max_val = -np.inf
    # One entry per sample: deltas[i] is the great-circle length of the leg
    # *arriving* at sample i (0.0 for the start).  It must stay exactly the
    # same length as `samples` or every distance statistic shifts by one.
    deltas: List[float] = []
    for i, (r, c) in enumerate(path):
        step_km = 0.0
        if i:
            la0, lo0 = math.radians(float(lat[path[i - 1][0]])), math.radians(float(lon[path[i - 1][1]]))
            la1, lo1 = math.radians(float(lat[r])), math.radians(float(lon[c]))
            x = math.cos(la0) * math.cos(lo0) * math.cos(la1) * math.cos(lo1) \
                + math.cos(la0) * math.sin(lo0) * math.cos(la1) * math.sin(lo1) \
                + math.sin(la0) * math.sin(la1)
            step_km = R_KM * math.acos(max(-1.0, min(1.0, x)))
            cum += step_km
        deltas.append(step_km)
        v = float(grid.sic_mean[r, c])
        v = v if math.isfinite(v) else None
        if v is not None and v > max_val:
            max_val, max_idx = v, i
        samples.append({
            "i": i,
            "row": r,
            "col": c,
            "lat": round(float(lat[r]), 4),
            "lon": round(float(lon[c]), 4),
            "cum_km": round(cum, 2),
            "delta_km": round(step_km, 3),
            "sic": None if v is None else round(v, 6),
            "band": _sic_band(v),
            "high_ice": None if v is None else bool(v >= SIC_HIGH_THRESHOLD),
        })

    # ---- real statistics over the route --------------------------------
    rows = [p[0] for p in path]
    cols = [p[1] for p in path]
    values = [s["sic"] for s in samples]
    finite_vals = [v for v in values if v is not None]

    band_cells = {name: 0 for name, _lo, _hi in SIC_BANDS}
    band_cells["invalid"] = 0
    band_km = {name: 0.0 for name, _lo, _hi in SIC_BANDS}
    band_km["invalid"] = 0.0
    high_cells = 0
    high_km = 0.0
    total_km = float(cum)
    for s, d in zip(samples, deltas):
        name = s["band"]
        band_cells[name] += 1
        band_km[name] += d
        if s["high_ice"]:
            high_cells += 1
            high_km += d

    def _pct(x: float) -> Optional[float]:
        return None if total_km <= 0 else round(100.0 * x / total_km, 2)

    # Distance-weighted mean SIC: only the legs whose SIC is a real number
    # take part, so a NaN cell can never be read as 0.
    valid_legs = [(s, d) for s, d in zip(samples, deltas) if s["sic"] is not None]
    valid_km = sum(d for _s, d in valid_legs)
    dw_mean = (sum((s["sic"] or 0.0) * d for s, d in valid_legs) / valid_km
               if valid_km > 0 else None)

    # ---- objective attribution (real, no modelling) --------------------
    env_sum = float(np.sum(cost_map.env_cost[rows, cols])) if path else 0.0
    sic_term = cost_map.terms.get("sic")
    sic_sum = float(np.sum(sic_term[rows, cols])) if (path and sic_term is not None) else 0.0
    distance_sum = float(result.route_length) * float(weights.w_distance)
    total_cost = float(result.total_cost)
    remainder = total_cost - env_sum - distance_sum

    cost_split = {
        "total_cost": _jsonable(total_cost),
        "environmental_cost": _jsonable(env_sum),
        "sic_cost": _jsonable(sic_sum),
        "distance_cost": _jsonable(distance_sum),
        "other_environmental_cost": _jsonable(env_sum - sic_sum),
        "unattributed": _jsonable(remainder),
        "sic_share_pct": _pct_cost(sic_sum, total_cost),
        "distance_share_pct": _pct_cost(distance_sum, total_cost),
        "formula": "w_sic*sic_mean + w_distance*grid_distance",
        "weights": weights.to_dict(),
        "terms_active": sorted(cost_map.terms),
        "note": "environmental_cost = sum of CostMap env over the routed "
                "cells (start cell included); distance_cost = w_distance * "
                "A* route length; unattributed is the residual and is 0 when "
                "only these two terms are enabled.",
    }

    return {
        "timestep": t,
        "success": bool(result.success),
        "date": str(field().dates[t])[:10],
        "source": "REAL A* over backend/cache/routing_sic_2026.npy",
        "coordinate_space": "WGS84 lat/lon; row/col index the 0.25 deg grid",
        "route_available": bool(path),
        "grid": {
            "n_rows": int(grid.n_rows),
            "n_cols": int(grid.n_cols),
            "resolution_deg": float(getattr(grid, "resolution_deg", 0.25)),
            "lat_range": [float(np.min(lat)), float(np.max(lat))],
            "lon_range": [float(np.min(lon)), float(np.max(lon))],
        },
        "waypoints": int(result.num_waypoints),
        "route_length_grid_units": _jsonable(result.route_length),
        "total_cost": _jsonable(total_cost),
        "great_circle_length_km": round(cum, 2),
        "samples": samples,
        "max_sic_index": max_idx,
        "max_sic": _jsonable(max(finite_vals)) if finite_vals else None,
        "min_sic": _jsonable(min(finite_vals)) if finite_vals else None,
        "mean_sic": _jsonable(float(np.mean(finite_vals))) if finite_vals else None,
        "mean_sic_distance_weighted": _jsonable(dw_mean),
        "nan_cells_on_route": sum(1 for v in values if v is None),
        "high_ice_threshold": SIC_HIGH_THRESHOLD,
        "high_ice_threshold_label": "moderate-pack onset "
                                    "(build_routing_grid.py band edge)",
        "high_ice_cells": high_cells,
        "high_ice_cells_pct": (None if not samples
                               else round(100.0 * high_cells / len(samples), 2)),
        "high_ice_km": round(high_km, 2),
        "high_ice_km_pct": _pct(high_km),
        "band_edges": [{"name": n, "lo": lo, "hi": hi} for n, lo, hi in SIC_BANDS],
        "bands": {
            name: {
                "cells": band_cells[name],
                "cells_pct": (None if not samples
                              else round(100.0 * band_cells[name] / len(samples), 2)),
                "km": round(band_km[name], 2),
                "km_pct": _pct(band_km[name]),
            }
            for name in list(band_cells)
        },
        "cost_split": cost_split,
        "note": "cum_km uses a great-circle approximation on the real 0.25 deg "
                "grid; route_length_grid_units is the repository's own A* cost.",
    }


def _pct_cost(part: float, whole: float) -> Optional[float]:
    if not math.isfinite(part) or not math.isfinite(whole) or whole == 0:
        return None
    return round(100.0 * part / whole, 2)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def create_app() -> Flask:
    app = Flask(__name__, static_folder=None)
    CORS(app)

    # ---------------- AURORA integration layer ----------------
    # The unified compositor (/api/sic/status, /api/sic/predict,
    # /api/aurora/status, /api/aurora/analyze) and the iceberg detector
    # blueprint.  Registered here so one app serves all three models; the
    # compositor reuses the helpers defined in this module rather than
    # reimplementing routing or raster encoding.
    from backend.api.aurora_api import bp as aurora_bp
    app.register_blueprint(aurora_bp)

    from backend.api.iceberg_api import bp as iceberg_bp
    app.register_blueprint(iceberg_bp)

    # ---------------- health ----------------

    @app.get("/api/health")
    def health():
        return jsonify({
            "status": "ok",
            "sic_artifact": (CACHE / "routing_sic_2026.npy").exists(),
            "route_artifact": ROUTE_JSON.exists(),
        })

    # ---------------- metadata ----------------

    @app.get("/api/sic/metadata")
    def sic_metadata():
        f = field()
        info = f.describe()
        spec = f.grid_spec()
        lat = np.load(str(CACHE / "routing_lat.npy"))
        lon = np.load(str(CACHE / "routing_lon.npy"))
        return jsonify({
            "project": "SIH2026059",
            "system": "IceRoute-Robust",
            "title": "Antarctic Ocean Route Optimization",
            "data_source": "REAL",
            "artifact": "backend/cache/routing_sic_2026.npy",
            "n_timesteps": info["n_time_steps"],
            "n_rows": spec["n_rows"],
            "n_cols": spec["n_cols"],
            "dtype": info["sic_dtype"],
            "resolution_deg": info["resolution_deg"],
            "lat_range": [spec["lat_min"], spec["lat_max"]],
            "lon_range": [spec["lon_min"], spec["lon_max"]],
            "lat": [round(float(v), 4) for v in lat],
            "lon": [round(float(v), 4) for v in lon],
            "lat_count": int(lat.size),
            "lon_count": int(lon.size),
            "model_band_rows": info["model_band_rows"],
            "model_band_cols": info["model_band_cols"],
            "extension_band_rows": info["extension_band_rows"],
            "lon_extension_cols": info["lon_extension_cols"],
            "date_range": info["date_range"],
            "dates": [str(d)[:10] for d in f.dates],
            "uncertainty_shape": info["uncertainty_shape"],
            "uncertainty_horizons": info["n_horizons"],
            "nan_policy": "NaN means INVALID / NON-NAVIGABLE. Never converted to "
                          "zero and never routed through.",
            "nan_encoding": {
                "format": "base64",
                "sic_u8": "uint8 SIC*255; NaN stored as 0 but meaningless",
                "valid": "packed bitmask, 1=valid measurement, 0=non-navigable",
                "authoritative_field": "valid",
            },
            "array_format_note": "GET /api/sic/<t>?format=array returns the "
                                 "field with explicit null for invalid cells.",
        })

    # ---------------- one SIC timestep ----------------

    @app.get("/api/sic/<int:timestep>")
    def sic_slice(timestep: int):
        check_timestep(timestep)
        sic2d = np.asarray(sic_mmap()[timestep], dtype=np.float64)
        payload: Dict[str, Any] = {
            "timestep": timestep,
            "day": timestep,
            "date": str(field().dates[timestep])[:10],
            "grid": {"n_rows": int(sic2d.shape[0]), "n_cols": int(sic2d.shape[1])},
            "stats": cached_stats(timestep),
            "provenance": "REAL (backend/cache/routing_sic_2026.npy)",
        }
        if request.args.get("format") == "array":
            payload["format"] = "array"
            payload["nan_representation"] = "null"
            payload["sic"] = encode_slice_array(sic2d)
        else:
            payload["format"] = "b64"
            payload["encoding"] = cached_slice_b64(timestep)
            payload["nan_encoding"] = {
                "sic_u8": "uint8 SIC*255, NaN written as 0",
                "valid": "packed bits, 1=valid, 0=NON-NAVIGABLE (authoritative)",
            }
        return jsonify(payload)

    # ---------------- verified route ----------------

    @app.get("/api/route")
    def verified_route():
        artifact = load_route_artifact()
        cells = artifact.get("final_route_cells")
        if not cells:
            return jsonify({"error": "route artifact has no final_route_cells"}), 500
        expert = artifact.get("expert_route", {})
        return jsonify({
            "source": "outputs/final_demo/final_route.json",
            "algorithm": "A* + CostMap (src/routing/astar.py, src/routing/cost.py)",
            "data": "REAL SIC",
            "success": bool(expert.get("waypoints", 0) > 0),
            "leg": artifact.get("leg", {}),
            "waypoints": int(expert.get("waypoints", 0)),
            "route_length_grid_units": expert.get("route_length_cells"),
            "total_cost": expert.get("total_cost"),
            "path": [[int(c[0]), int(c[1])] for c in cells],
            "dynamic_rerouting": artifact.get("dynamic_rerouting", {}),
            "limitations": LIMITATIONS,
        })

    @app.get("/api/route/at/<int:timestep>")
    def route_at(timestep: int):
        check_timestep(timestep)
        a = request.args
        plan = plan_route(
            timestep,
            int(a.get("start_row", DEFAULT_START[0])),
            int(a.get("start_col", DEFAULT_START[1])),
            int(a.get("goal_row", DEFAULT_GOAL[0])),
            int(a.get("goal_col", DEFAULT_GOAL[1])),
            _weights_key(),
        )

        return jsonify({
            "date": str(field().dates[timestep])[:10],
            "algorithm": "A* + CostMap",
            "data": "REAL SIC",
            "nan_policy": "NaN cells excluded from routing (non-navigable)",
            **plan,
        })

    # ---------------- interactive optimization (POST) ----------------

    @app.post("/api/route/optimize")
    def route_optimize():
        """
        Plan a route from geographic coordinates against the real SIC field.

        Body: {start_lat, start_lon, goal_lat, goal_lon, timestep}
        Optional: {snap: bool = true, max_snap_cells: int = 3}

        Rejects (HTTP 400, never a fabricated route) out-of-grid coordinates,
        non-finite values, endpoints with no navigable cell in range, and
        unreachable goals.  A returned route is always proven in-bounds,
        8-connected, goal-reaching and free of NaN / non-navigable cells.
        """
        body = request.get_json(silent=True) or {}
        if not isinstance(body, dict):
            raise RouteRequestError("request body must be a JSON object",
                                    "invalid_body")

        missing = [k for k in ("start_lat", "start_lon", "goal_lat",
                               "goal_lon", "timestep") if body.get(k) is None]
        if missing:
            raise RouteRequestError(
                f"missing required field(s): {', '.join(missing)}",
                "missing_fields", missing=missing)

        try:
            max_snap = int(body.get("max_snap_cells", MAX_SNAP_CELLS))
        except (TypeError, ValueError):
            raise RouteRequestError("max_snap_cells must be an integer",
                                    "invalid_snap_radius")
        if not 0 <= max_snap <= 10:
            raise RouteRequestError(
                "max_snap_cells must be between 0 and 10", "invalid_snap_radius")

        snap = body.get("snap", True)
        if not isinstance(snap, bool):
            raise RouteRequestError("snap must be a boolean", "invalid_snap")

        return jsonify(optimize_route(
            start_lat=body["start_lat"], start_lon=body["start_lon"],
            goal_lat=body["goal_lat"], goal_lon=body["goal_lon"],
            timestep=body["timestep"], allow_snap=snap, max_snap_cells=max_snap,
        ))

    @app.post("/api/route/reroute")
    def route_reroute():
        """
        Re-optimize an existing route on a LATER real SIC forecast timestep.

        Body: {original_route: {path, start, goal, timestep, mean_sic, max_sic},
               new_timestep: int, start?: {...}, goal?: {...}, snap?: bool}

        Returns both routes, a like-for-like comparison and the real changed
        segments. Endpoints default to the original route's own endpoints, so
        the only thing that changes is the environment.
        """
        body = request.get_json(silent=True) or {}
        if not isinstance(body, dict):
            raise RouteRequestError("request body must be a JSON object",
                                    "invalid_body")
        if body.get("new_timestep") is None:
            raise RouteRequestError("missing required field: new_timestep",
                                    "missing_fields",
                                    missing=["new_timestep"])
        snap = body.get("snap", True)
        if not isinstance(snap, bool):
            raise RouteRequestError("snap must be a boolean", "invalid_snap")
        return jsonify(reroute_route(
            original=body.get("original_route"),
            new_timestep=body["new_timestep"],
            start=body.get("start"), goal=body.get("goal"), allow_snap=snap,
        ))

    # ---------------- dynamic rerouting ----------------

    @app.get("/api/reroute/<int:timestep>")
    def reroute(timestep: int):
        a = request.args
        origin = int(a.get("origin_timestep", 0))
        check_timestep(timestep)
        check_timestep(origin)
        sr = int(a.get("start_row", DEFAULT_START[0]))
        sc = int(a.get("start_col", DEFAULT_START[1]))
        gr = int(a.get("goal_row", DEFAULT_GOAL[0]))
        gc = int(a.get("goal_col", DEFAULT_GOAL[1]))

        original = plan_route(origin, sr, sc, gr, gc, _weights_key())
        new = plan_route(timestep, sr, sc, gr, gc, _weights_key())
        orig_cells = [tuple(p) for p in original["path"]]
        new_cells = [tuple(p) for p in new["path"]]
        both = bool(orig_cells and new_cells)

        return jsonify({
            "origin_timestep": origin,
            "reroute_timestep": timestep,
            "forecast_step_days": int(timestep - origin),
            "origin_date": str(field().dates[origin])[:10],
            "reroute_date": str(field().dates[timestep])[:10],
            "status": "SUCCESS" if new["success"] else "FAILED",
            "algorithm": "A* + CostMap on real SIC (src/routing/astar.py)",
            "reroute_logic": "src/routing/scenario_router.jaccard_overlap / "
                             "route_coverage",
            "original_route": {
                "success": original["success"],
                "waypoints": original["waypoints"],
                "route_length_grid_units": original["route_length_grid_units"],
                "mean_sic": original["mean_sic"],
                "max_sic": original["max_sic"],
                "nan_cells_on_route": original["nan_cells_on_route"],
                "path": original["path"],
            },
            "rerouted_route": {
                "success": new["success"],
                "waypoints": new["waypoints"],
                "route_length_grid_units": new["route_length_grid_units"],
                "mean_sic": new["mean_sic"],
                "max_sic": new["max_sic"],
                "nan_cells_on_route": new["nan_cells_on_route"],
                "path": new["path"],
            },
            "comparison": {
                "jaccard_overlap": float(jaccard_overlap(orig_cells, new_cells))
                if both else None,
                "route_coverage": float(route_coverage(orig_cells, new_cells))
                if both else None,
                "changed_cells": len(set(orig_cells) ^ set(new_cells))
                if both else None,
                "waypoints_before": original["waypoints"],
                "waypoints_after": new["waypoints"],
            },
            "limitations": LIMITATIONS,
        })

    @app.get("/api/map/coastline")
    def map_coastline():
        """Real Antarctic coastline geometry for the basemap layer."""
        return jsonify(coastline())

    @app.get("/api/limitations")
    def limitations():
        return jsonify(LIMITATIONS)

    @app.get("/api/layers/status")
    def layers_status():
        """
        Per-layer availability for a given forecast timestep.

        Reports what each environmental layer contributed to the grid the route
        optimizer consumes, and whether that layer actually reached the cost.
        A layer is never reported as integrated on the strength of a loader
        existing: ``in_cost`` is derived from the active cost weights.
        """
        t = int(request.args.get("timestep", 0))
        check_timestep(t)
        grid, cost_map, report = route_grid(t, _weights_key())
        return jsonify({
            "timestep": t,
            "date": str(field().dates[t])[:10],
            "environmental_grid": report["grid"],
            "land_mask": report["land_mask"],
            "cost_weights": cost_weights().to_dict(),
            "cost_breakdown": report["cost_breakdown"],
            **registry_to_dict(
                report["registry"],
                cost_layers=report["cost_breakdown"]["layers_in_cost"]),
            "datasets": data_paths.describe_all(),
            "data_root_env": data_paths.DATA_ROOT_ENVVAR,
            "limitations": LIMITATIONS,
        })


    # ---------------- system status / data availability ----------------

    @app.get("/api/system/status")
    def system_status():
        """What is genuinely available for this run. No optimistic claims."""
        sic_mmap()  # raises 503 if the real artifact is missing
        f = field()
        info = f.describe()
        spec = f.grid_spec()

        # --- CMEMS: discover, never fabricate ---
        cmems_available, cmems_note, cmems_files = discover_cmems()
        # --- SIC forecasting checkpoints (teammate's ConvLSTM) ---
        ckpts = discover_sic_checkpoints()
        # --- raw inference inputs ---
        raw_inputs = [p for p in (
            ROOT / "backend" / "data" / "test_2026",
            ROOT / "backend" / "data" / "processed",
            DATA_ROOT / "dataset" / "test_2026",
        ) if p.is_dir() and any(p.iterdir())]
        # --- iceberg / CVaR inputs ---
        iceberg = discover_iceberg()
        # --- route ML policy ---
        policy = discover_route_policy()

        return jsonify({
            "project": "SIH2026059",
            "system": "IceRoute-Robust",
            "title": "Antarctic Ocean Route Optimization",
            "data_root": str(DATA_ROOT),
            "dataset_location": DATASET_LOCATION,
            "datasets": data_paths.describe_all(),
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "environment": {
                "sic": {
                    "available": True,
                    "label": "Committed 2026 SIC forecast output",
                    "artifact": "backend/cache/routing_sic_2026.npy",
                    "n_timesteps": info["n_time_steps"],
                    "grid": [spec["n_rows"], spec["n_cols"]],
                    "resolution_deg": spec["resolution_deg"],
                    "lat_range": [spec["lat_min"], spec["lat_max"]],
                    "lon_range": [spec["lon_min"], spec["lon_max"]],
                    "date_range": info["date_range"],
                    "uncertainty_shape": info["uncertainty_shape"],
                    "uncertainty_horizons": info["n_horizons"],
                    "nan_policy": "NaN = non-navigable; never zero-filled",
                },
                "cmems": {
                    "available": cmems_available,
                    "label": ("CMEMS current data available"
                              if cmems_available else
                              "CMEMS CURRENT DATA UNAVAILABLE"),
                    "root": cmems_note,
                    "sample_files": cmems_files[:5],
                    "variables": ["uo", "vo"] if cmems_available else [],
                    "integration_present": True,
                    "note": ("Currents were NOT part of the route for this run; "
                             "no currents are faked."
                             if not cmems_available else
                             "Currents available; current_cost/uo/vo populated "
                             "on the EnvironmentalGrid."),
                },
                "iceberg": iceberg,
                "cvar": {
                    "available": bool(iceberg.get("available")),
                    "label": ("CVaR available" if iceberg.get("available") else
                              "CVaR unavailable \u2014 iceberg risk/uncertainty "
                              "data not available"),
                    "computed": False,
                    "note": "No CVaR value is computed or claimed in this demo.",
                },
            },
            "models": {
                "sic_forecaster": {
                    "present": any(c["is_sic_forecaster"] for c in ckpts),
                    "checkpoints": ckpts,
                    "role": "upstream inference component (teammate-owned)",
                    "inference_rerun_possible": bool(raw_inputs),
                    "label": ("Raw 2026 inference inputs present"
                              if raw_inputs else
                              "Committed 2026 SIC forecast output (raw inputs "
                              "absent; inference not re-run)"),
                    "raw_input_dirs": [str(p) for p in raw_inputs],
                },
                "route_policy": policy,
            },
            "routing": {
                "algorithm": "A* + CostMap (src/routing/astar.py, cost.py)",
                "safety": "NaN SIC cells marked non-navigable",
                "reroute": "A* replan on later real SIC forecast + "
                           "jaccard_overlap / route_coverage comparison",
                "verified_artifact": "outputs/final_demo/final_route.json",
            },
            "retraining_performed": False,
            "synthetic_route_data_used": False,
            "limitations": LIMITATIONS,
        })

    @app.get("/api/current/<int:timestep>")
    def current_slice(timestep: int):
        """
        CMEMS uo/vo for the requested forecast date, if real CMEMS data is
        reachable from the configured data root.  Never fabricates currents.
        """
        check_timestep(timestep)
        available, note, _files = discover_cmems()
        payload: Dict[str, Any] = {
            "timestep": timestep,
            "date": str(field().dates[timestep])[:10],
            "available": available,
            "label": ("CMEMS current data available" if available
                      else "CMEMS CURRENT DATA UNAVAILABLE"),
            "root": note,
            "integration": "src/data/cmems_loader.py (CMEMSDateAwareLoader)",
        }
        if not available:
            payload["uo"] = None
            payload["vo"] = None
            payload["note"] = ("No currents are returned. The UI must not show "
                               "current arrows and the route for this run did "
                               "not include current cost.")
            return jsonify(payload)

        # Real data path: reuse the existing loader (no reimplementation).
        try:
            from src.data.cmems_loader import CMEMSDateAwareLoader
            query_dt = datetime.combine(
                datetime.fromisoformat(payload["date"]), datetime.min.time(),
                tzinfo=timezone.utc,
            )
            loader = CMEMSDateAwareLoader(cmems_root=CMEMS_DATE_AWARE_ROOT,
                                          route_start_datetime=query_dt)
            uo, vo, _lat, _lon = loader.load_uo_vo(t_hours=0.0)
            payload.update({
                "uo": encode_slice_b64(np.asarray(uo, dtype=np.float64)),
                "vo": encode_slice_b64(np.asarray(vo, dtype=np.float64)),
                "grid": {"n_rows": int(uo.shape[0]), "n_cols": int(uo.shape[1])},
                "note": "Real CMEMS GLORYS surface currents.",
            })
        except Exception as exc:  # honest failure, never fake values
            payload.update({"available": False,
                            "label": "CMEMS CURRENT DATA UNAVAILABLE",
                            "uo": None, "vo": None,
                            "note": f"{type(exc).__name__}: {exc}"})
        return jsonify(payload)

    # ---------------- forecast uncertainty (real artifact) ----------------

    @app.get("/api/uncertainty/summary")
    def uncertainty_summary():
        return jsonify(discover_uncertainty_summary())

    @app.get("/api/uncertainty/<int:timestep>")
    def uncertainty_slice(timestep: int):
        """
        One forecast-uncertainty frame from the committed
        ``uncertainty_2026.npy`` artifact, expanded to the full routing grid
        and encoded with the same value/validity contract as the SIC endpoint.

        The artifact's 3 channels are lead-time horizons; no unit is invented
        beyond the SIC fraction itself.
        """
        horizon = int(request.args.get("horizon", 0))
        frame = uncertainty_frame(timestep, horizon)
        st = uncertainty_stats(frame)
        return jsonify({
            "timestep": timestep,
            "date": str(field().dates[timestep])[:10],
            "horizon": horizon,
            "horizon_days": horizon + 1,
            "n_horizons": int(uncertainty_mmap().shape[1]),
            "source": "backend/cache/uncertainty_2026.npy",
            "model_band_rows": [0, int(uncertainty_mmap().shape[2])],
            "model_band_cols": [0, int(uncertainty_mmap().shape[3])],
            "interpretation": "Higher value = less confidence in the forecast "
                              "SIC for that cell at this lead time.",
            "quantity": "SIC forecast spread (SIC fraction, 0-1); the artifact "
                        "carries no physical unit beyond the SIC scale",
            "stats": st,
            "encoding": encode_slice_b64(frame),
        })

    # ---------------- SIC forecast ensemble (checkpoints) ----------------

    @app.get("/api/models/ensemble")
    def models_ensemble():
        return jsonify(discover_ensemble())

    # ---------------- route risk profile ----------------

    @app.get("/api/route/profile/<int:timestep>")
    def route_profile_endpoint(timestep: int):
        """
        Real SIC encountered along the real A* route, one sample per
        waypoint, with cumulative great-circle distance.
        """
        sr = int(request.args.get("start_row", DEFAULT_START[0]))
        sc = int(request.args.get("start_col", DEFAULT_START[1]))
        gr = int(request.args.get("goal_row", DEFAULT_GOAL[0]))
        gc = int(request.args.get("goal_col", DEFAULT_GOAL[1]))
        check_timestep(timestep)
        return jsonify(route_profile(timestep, sr, sc, gr, gc))

    # ---------------- built React app ----------------

    @app.get("/")
    def index():
        if FRONTEND_DIST.is_dir() and (FRONTEND_DIST / "index.html").exists():
            return send_from_directory(str(FRONTEND_DIST), "index.html")
        return jsonify({
            "status": "backend running",
            "hint": "build the React app with: cd frontend/aurora && "
                    "npm install && npm run build  (or use npm run dev)",
            "endpoints": ["/api/health", "/api/sic/metadata",
                          "/api/sic/<timestep>", "/api/route",
                          "/api/route/at/<timestep>",
                          "POST /api/route/optimize",
                          "/api/route/profile/<timestep>",
                          "/api/uncertainty/<timestep>?horizon=0..2",
                          "/api/uncertainty/summary",
                          "/api/models/ensemble",
                          "/api/reroute/<timestep>", "/api/limitations"],
        })

    @app.get("/<path:filename>")
    def static_files(filename: str):
        if filename.startswith("api/"):
            return jsonify({"error": "not found"}), 404
        if FRONTEND_DIST.is_dir():
            candidate = FRONTEND_DIST / filename
            if candidate.is_file():
                return send_from_directory(str(FRONTEND_DIST), filename)
            index_path = FRONTEND_DIST / "index.html"
            if index_path.exists():
                return send_from_directory(str(FRONTEND_DIST), "index.html")
        return jsonify({"error": "not found"}), 404

    @app.errorhandler(404)
    def not_found(_exc):
        return jsonify({"error": "not found"}), 404

    @app.errorhandler(IndexError)
    def bad_index(exc):
        return jsonify({"error": str(exc)}), 404

    @app.errorhandler(RouteRequestError)
    def bad_route_request(exc):
        return jsonify(exc.to_payload()), 400

    @app.errorhandler(FileNotFoundError)
    def missing_file(exc):
        return jsonify({"error": str(exc)}), 503

    return app


app = create_app()


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="SIH2026059 read-only SIC API")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    print(f"SIH2026059 IceRoute-Robust API  ->  http://{args.host}:{args.port}")
    print(f"  real SIC : {CACHE / 'routing_sic_2026.npy'}")
    print(f"  route    : {ROUTE_JSON}")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
