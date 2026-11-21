"""Reconstruct the Wiegel et al. (2021) screening method, not measured port downtime.

Reference: https://doi.org/10.3390/jmse9070695 (CC BY 4.0), especially section 2
and Appendix A. Numerical spectrum choices and deviations are recorded in the
companion notebook. No digitized scatter values enter these calculations.
"""

from __future__ import annotations

import calendar
import hashlib
import json
import logging
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
import xarray as xr
from scipy import integrate
from scipy.spatial import cKDTree
from pyproj import Geod
from shapely.geometry import LineString, Point
from shapely.ops import linemerge, unary_union
from shapely.prepared import prep

LOGGER = logging.getLogger(__name__)
GRAVITY = 9.81
ERA5_DATASET = "reanalysis-era5-single-levels"
ERA5_VARIABLES = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "significant_height_of_combined_wind_waves_and_swell",
    "mean_wave_direction",
    "mean_wave_period",
    "peak_wave_period",
]
NATURAL_EARTH_BASE = "https://naturalearth.s3.amazonaws.com/10m_physical"
ALIASES = {
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "significant_height_of_combined_wind_waves_and_swell": "swh",
    "mean_wave_direction": "mwd",
    "mean_wave_period": "mwp",
    "peak_wave_period": "pp1d",
    "model_bathymetry": "wmb",
}


@dataclass(frozen=True)
class Scenario:
    name: str = "baseline"
    depth_m: float = 20.0
    wind_limit: float = 13.8
    wave_limit: float = 2.0
    ig_limit: float = 0.05
    wind_factor: float = 1.0
    wave_factor: float = 1.0
    gap_hours: int = 4

    def __post_init__(self):
        values = [self.depth_m, self.wind_limit, self.wave_limit, self.ig_limit,
                  self.wind_factor, self.wave_factor]
        if not np.isfinite(values).all() or min(values) <= 0 or self.gap_hours < 0:
            raise ValueError("Depths, thresholds and scale factors must be positive; gap >= 0.")


SCENARIOS = [
    Scenario(),
    Scenario("lower_thresholds", wind_limit=10.8, wave_limit=1.5, ig_limit=0.04),
    Scenario("higher_thresholds", wind_limit=17.2, wave_limit=2.5, ig_limit=0.06),
    Scenario("depth_17m", depth_m=17),
    Scenario("depth_16m", depth_m=16),
    Scenario("depth_15m", depth_m=15),
    Scenario("wind_plus_5pct", wind_factor=1.05),
    Scenario("waves_plus_5pct", wave_factor=1.05),
    Scenario("wind_waves_plus_5pct", wind_factor=1.05, wave_factor=1.05),
]

# Approximate geographic seed coordinates, NOT the authors' missing WPI-2019 selection.
VALIDATION_PORTS = pd.DataFrame([
    ("Doraleh", 43.08, 11.60, False, 99.9),
    ("Jeddah", 39.16, 21.48, False, 99.8),
    ("Hay Point", 149.30, -21.28, False, 99.7),
    ("Shanghai", 121.60, 31.35, False, 99.4),
    ("Hadera", 34.87, 32.47, False, 96.2),
    ("Long Beach", -118.22, 33.75, True, 95.9),
    ("Tomakomai", 141.62, 42.63, True, 94.9),
    ("Salalah", 54.00, 16.95, True, 87.8),
    ("Cape Town", 18.43, -33.91, True, 54.1),
    ("Geraldton", 114.60, -28.78, True, 46.2),
], columns=["name", "port_lon", "port_lat", "known_issues", "published_operability"])


def data_directories(root: Path = Path("data")) -> dict[str, Path]:
    directories = {name: Path(root) / name for name in
                   ("raw", "reference", "processed", "figures")}
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)
    return directories


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_reference(url: str, target: Path) -> Path:
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    with requests.get(url, stream=True, timeout=(30, 180)) as response:
        response.raise_for_status()
        with partial.open("wb") as stream:
            for block in response.iter_content(1024 * 1024):
                stream.write(block)
    partial.replace(target)
    write_json(target.with_suffix(target.suffix + ".source.json"), {
        "url": url, "downloaded_utc": datetime.now(timezone.utc).isoformat(),
        "sha256": file_hash(target), "bytes": target.stat().st_size,
    })
    return target


def retrieve_era5(request: dict, target: Path) -> Path:
    """Cache by complete public request; never serialize a client or credentials."""
    manifest = target.with_suffix(target.suffix + ".request.json")
    if target.exists():
        if not manifest.exists() or json.loads(manifest.read_text())["request"] != request:
            raise ValueError(f"Cached request differs or has no provenance: {target}")
        return target
    import cdsapi

    target.parent.mkdir(parents=True, exist_ok=True)
    write_json(manifest, {"dataset": ERA5_DATASET, "request": request})
    client = cdsapi.Client(quiet=False, debug=False, progress=False, retry_max=3, timeout=60)
    partial = target.with_suffix(target.suffix + ".part")
    client.retrieve(ERA5_DATASET, request, str(partial))
    partial.replace(target)
    metadata = json.loads(manifest.read_text())
    metadata.update(sha256=file_hash(target), bytes=target.stat().st_size)
    write_json(manifest, metadata)
    return target


def era5_request(year: int, months: list[int], area: list[float],
                 variables: Optional[list[str]] = None, snapshot: bool = False) -> dict:
    return {
        "product_type": ["reanalysis"],
        "variable": variables or ERA5_VARIABLES,
        "year": [str(year)],
        "month": [f"{m:02d}" for m in months],
        "day": ["01"] if snapshot else [f"{d:02d}" for d in range(1, 32)],
        "time": ["00:00"] if snapshot else [f"{h:02d}:00" for h in range(24)],
        "area": area, "grid": [0.5, 0.5],
        "data_format": "netcdf", "download_format": "unarchived",
    }


def open_era5(path: Path) -> xr.Dataset:
    """Handle CDS' NetCDF or multi-stream ZIP responses without trusting ZIP paths."""
    if zipfile.is_zipfile(path):
        folder = path.with_suffix(path.suffix + ".contents")
        folder.mkdir(exist_ok=True)
        files = []
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                if member.is_dir() or not member.filename.endswith(".nc"):
                    continue
                destination = folder / Path(member.filename).name
                with archive.open(member) as source, destination.open("wb") as sink:
                    import shutil
                    shutil.copyfileobj(source, sink)
                files.append(destination)
        if not files:
            raise ValueError(f"CDS archive contains no NetCDF files: {path}")
    else:
        files = [path]
    parts = []
    for file in files:
        with xr.open_dataset(file, engine="netcdf4") as source:
            ds = source.load()
        renames = {k: v for k, v in ALIASES.items() if k in ds}
        if "valid_time" in ds.dims:
            renames["valid_time"] = "time"
        ds = ds.rename(renames)
        for coordinate in ("expver", "number", "surface"):
            if coordinate in ds.dims and ds.sizes[coordinate] == 1:
                ds = ds.isel({coordinate: 0}, drop=True)
        if "longitude" in ds.coords:
            ds = ds.assign_coords(longitude=((ds.longitude + 180) % 360) - 180)
            ds = ds.sortby("longitude")
        parts.append(ds)
    return xr.merge(parts, compat="no_conflicts", join="outer")


def download_coastline(root: Path = Path("data")):
    import geopandas as gpd

    reference = data_directories(root)["reference"]
    coast = download_reference(f"{NATURAL_EARTH_BASE}/ne_10m_coastline.zip",
                               reference / "ne_10m_coastline.zip")
    land = download_reference(f"{NATURAL_EARTH_BASE}/ne_10m_land.zip",
                              reference / "ne_10m_land.zip")
    return gpd.read_file(coast), gpd.read_file(land)


def _line_parts(coast) -> list[np.ndarray]:
    parts = []
    for geometry in coast.geometry:
        geometries = geometry.geoms if geometry.geom_type == "MultiLineString" else [geometry]
        for line in geometries:
            coordinates = np.asarray(line.coords)[:, :2]
            if len(coordinates) >= 9:
                parts.append(coordinates)
    if not parts:
        raise ValueError("No coastline segments with enough vertices.")
    return parts


def coastal_grid_candidates(snapshot: xr.Dataset, coast, land) -> pd.DataFrame:
    """All wet native cells in the paper's coastal band, before coastline sampling."""
    if "wmb" not in snapshot:
        raise ValueError("Global model bathymetry is required to select ocean grid points.")
    bathymetry = snapshot.wmb
    if "time" in bathymetry.dims:
        bathymetry = bathymetry.isel(time=0)
    wet = bathymetry.to_dataframe(name="depth0").reset_index()
    wet = wet.loc[np.isfinite(wet.depth0) & (wet.depth0 > 0),
                  ["longitude", "latitude", "depth0"]].copy()
    parts = _line_parts(coast)
    vertices = np.concatenate(parts)
    # Longitude copies make the degree-distance search continuous at the dateline.
    extended = np.concatenate([vertices + [shift, 0] for shift in (-360, 0, 360)])
    tree = cKDTree(extended)
    distance, _ = tree.query(wet[["longitude", "latitude"]].to_numpy())
    wet = wet.loc[(distance >= 0.1) & (distance <= 0.8)].copy()
    wet["distance_to_coast_deg"] = distance[(distance >= 0.1) & (distance <= 0.8)]
    land_union = prep(unary_union(land.geometry))
    outside_land = [not land_union.covers(Point(lon, lat))
                    for lon, lat in wet[["longitude", "latitude"]].to_numpy()]
    wet = wet.loc[outside_land].rename(columns={"longitude": "lon", "latitude": "lat"})
    return wet.reset_index(drop=True)


def build_coastal_grid(snapshot: xr.Dataset, coast, land) -> pd.DataFrame:
    """Reconstruct a grid, without claiming to reproduce unpublished manual exclusions."""
    wet = coastal_grid_candidates(snapshot, coast, land)
    parts = _line_parts(coast)
    vertices = np.concatenate(parts)
    eligible = wet[["lon", "lat"]].to_numpy()
    if not len(eligible):
        raise ValueError("No eligible ocean grid points in the coastal distance band.")
    copies = np.concatenate([eligible + [shift, 0] for shift in (-360, 0, 360)])
    nearest_distance, nearest_index = cKDTree(copies).query(vertices)
    chosen = np.unique(nearest_index[nearest_distance <= 0.8] % len(eligible))
    wet = wet.iloc[chosen].copy()
    normals = shoreline_normals(wet[["lon", "lat"]].to_numpy(), parts)
    wet["shore_normal"] = normals
    wet = wet.sort_values(["lat", "lon"]).reset_index(drop=True)
    wet.insert(0, "site_id", np.arange(len(wet)))
    return wet


def shoreline_normals(points: np.ndarray, parts: list[np.ndarray]) -> np.ndarray:
    vertices = np.concatenate(parts)
    segment_ids = np.concatenate([np.full(len(part), i) for i, part in enumerate(parts)])
    vertex_ids = np.concatenate([np.arange(len(part)) for part in parts])
    extended = np.concatenate([vertices + [shift, 0] for shift in (-360, 0, 360)])
    _, indices = cKDTree(extended).query(points)
    indices %= len(vertices)
    normals = []
    for (lon, lat), index in zip(points, indices):
        part = parts[segment_ids[index]]
        j = vertex_ids[index]
        if np.allclose(part[0], part[-1]):
            ring = part[:-1]
            left = ring[(j - np.arange(1, 5)) % len(ring)].mean(axis=0)
            right = ring[(j + np.arange(1, 5)) % len(ring)].mean(axis=0)
        else:
            left = part[max(0, j - 4):j + 1].mean(axis=0)
            right = part[j:min(len(part), j + 5)].mean(axis=0)
        tangent = right - left
        tangent[0] = (tangent[0] + 180) % 360 - 180
        tangent[0] *= np.cos(np.deg2rad(lat))
        normal = np.array([tangent[1], -tangent[0]])
        offshore = np.array([((lon - vertices[index, 0] + 180) % 360 - 180)
                             * np.cos(np.deg2rad(lat)), lat - vertices[index, 1]])
        if np.linalg.norm(normal) == 0:
            raise ValueError("Degenerate coastline tangent; supply a reviewed site normal.")
        if normal @ offshore < 0:
            normal *= -1
        normals.append(np.rad2deg(np.arctan2(normal[0], normal[1])) % 360)
    return np.asarray(normals)


def _sphere_coordinates(coordinates: np.ndarray) -> np.ndarray:
    lon, lat = np.deg2rad(coordinates).T
    return np.column_stack((np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)))


def _coverage_distance_km(coordinates: np.ndarray, reference: np.ndarray) -> np.ndarray:
    _, nearest = cKDTree(_sphere_coordinates(reference)).query(_sphere_coordinates(coordinates))
    _, _, distance = Geod(ellps="WGS84").inv(
        coordinates[:, 0], coordinates[:, 1], reference[nearest, 0], reference[nearest, 1])
    return np.asarray(distance) / 1000


def _coastline_anchors(coast, spacing_km: float, simplify_degrees: float) -> pd.DataFrame:
    geod = Geod(ellps="WGS84")
    pieces = [part for geometry in coast.geometry
              for part in (geometry.geoms if geometry.geom_type == "MultiLineString" else [geometry])]
    merged = linemerge(pieces)
    lines = merged.geoms if merged.geom_type == "MultiLineString" else [merged]
    anchors = []
    for segment_id, line in enumerate(lines):
        coordinates = np.array(line.coords)[:, :2]
        # Unwrap before simplification so an antimeridian crossing is not a world-spanning chord.
        coordinates[:, 0] = np.rad2deg(np.unwrap(np.deg2rad(coordinates[:, 0])))
        coordinates = np.asarray(LineString(coordinates).simplify(simplify_degrees).coords)
        azimuth, _, distance = geod.inv(
            coordinates[:-1, 0].tolist(), coordinates[:-1, 1].tolist(),
            coordinates[1:, 0].tolist(), coordinates[1:, 1].tolist())
        azimuth, distance = np.asarray(azimuth), np.asarray(distance)
        valid = distance > 0
        starts, azimuth, distance = coordinates[:-1][valid], azimuth[valid], distance[valid]
        cumulative = np.r_[0, np.cumsum(distance) / 1000]
        length = cumulative[-1]
        closed = np.allclose(coordinates[0], coordinates[-1], rtol=0, atol=1e-8)
        if length < spacing_km * (2 if closed else 1):
            continue
        intervals = max(2 if closed else 1, round(length / spacing_km))
        step = length / intervals
        stations = ((np.arange(intervals) + 0.5) * step if closed
                    else np.linspace(0, length, intervals + 1))
        edges = np.minimum(np.searchsorted(cumulative, stations, side="right") - 1,
                           len(distance) - 1)
        lon, lat, _ = geod.fwd(starts[edges, 0], starts[edges, 1], azimuth[edges],
                              (stations - cumulative[edges]) * 1000)
        anchors.append(pd.DataFrame({
            "coast_segment": segment_id, "along_coast_km": stations,
            "anchor_lon": (lon + 180) % 360 - 180, "anchor_lat": lat,
            "anchor_step_km": step,
        }))
    if not anchors:
        raise ValueError("No coastline segments long enough for the requested spacing.")
    return pd.concat(anchors, ignore_index=True)


def regularise_coastal_grid(candidates: pd.DataFrame, coast, coverage: pd.DataFrame,
                           excluded_land=(), spacing_km: float = 55.6,
                           coverage_radius_km: float = 150,
                           simplify_degrees: float = 0.1) -> pd.DataFrame:
    """Sample one coastline track, then snap to unique eligible native ERA5 nodes.

    Coverage is an explicitly approximate spatial mask, not a reconstruction of
    missing original records. No recovered colours or operability values are used.
    """
    settings = [spacing_km, coverage_radius_km, simplify_degrees]
    if not np.isfinite(settings).all() or min(settings[:2]) <= 0 or simplify_degrees < 0:
        raise ValueError("Spacing/radius must be positive and simplification nonnegative.")
    if candidates.empty or coverage.empty or candidates.duplicated(["lon", "lat"]).any():
        raise ValueError("Nonempty unique ERA5 candidates and nonempty coverage points required.")
    coordinates = candidates[["lon", "lat"]].to_numpy(float)
    reference = coverage[["lon", "lat"]].to_numpy(float)
    if (not np.isfinite(coordinates).all() or not np.isfinite(reference).all()
            or not np.allclose(coordinates * 2, np.round(coordinates * 2), rtol=0, atol=1e-8)):
        raise ValueError("Candidates must lie on the finite native 0.5-degree ERA5 grid.")
    if not candidates.distance_to_coast_deg.between(0.1, 0.8).all():
        raise ValueError("Candidate coastal distances must be in the paper's 0.1-0.8 degree band.")
    exclusion = prep(unary_union(excluded_land).buffer(0.8)) if len(excluded_land) else None
    coverage_distance = _coverage_distance_km(coordinates, reference)
    keep = coverage_distance <= coverage_radius_km
    if exclusion is not None:
        keep &= [not exclusion.covers(Point(lon, lat)) for lon, lat in coordinates]
    candidates = candidates.loc[keep].copy()
    candidates["coverage_distance_km"] = coverage_distance[keep]
    if candidates.empty:
        raise ValueError("No eligible ERA5 candidates within the reference coverage.")
    anchors = _coastline_anchors(coast, spacing_km, simplify_degrees)
    anchor_coordinates = anchors[["anchor_lon", "anchor_lat"]].to_numpy()
    keep = _coverage_distance_km(anchor_coordinates, reference) <= coverage_radius_km
    if exclusion is not None:
        keep &= [not exclusion.covers(Point(lon, lat)) for lon, lat in anchor_coordinates]
    anchors = anchors.loc[keep].copy()
    if anchors.empty:
        raise ValueError("No coastline anchors within the reference coverage.")
    coordinates = anchors[["anchor_lon", "anchor_lat"]].to_numpy()
    extended = np.concatenate([coordinates + [shift, 0] for shift in (-360, 0, 360)])
    distance, nearest = cKDTree(extended).query(candidates[["lon", "lat"]])
    selected = candidates.loc[distance <= 0.8].reset_index(drop=True)
    anchor_index = nearest[distance <= 0.8] % len(anchors)
    anchors = anchors.iloc[anchor_index].reset_index(drop=True)
    if selected.empty:
        raise ValueError("No eligible ERA5 nodes within 0.8 degrees of coastline anchors.")
    _, _, snap_distance = Geod(ellps="WGS84").inv(
        anchors.anchor_lon.to_numpy(), anchors.anchor_lat.to_numpy(),
        selected.lon.to_numpy(), selected.lat.to_numpy())
    result = pd.concat([selected, anchors], axis=1)
    result["anchor_index"] = anchor_index
    result["snap_distance_km"] = snap_distance / 1000
    # Assign cells to anchors first: anchor-to-cell snapping alone can skip valid intervening cells.
    result = (result.sort_values("snap_distance_km", kind="stable")
              .drop_duplicates("anchor_index")
              .sort_values(["coast_segment", "along_coast_km"]).reset_index(drop=True))
    return result


def select_port_sites(grid: pd.DataFrame,
                      ports: pd.DataFrame = VALIDATION_PORTS) -> pd.DataFrame:
    """Map reviewed seed coordinates to nearest reconstructed wet coastal sites."""
    lon, lat = np.deg2rad(grid[["lon", "lat"]].to_numpy()).T
    sphere = np.column_stack((np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)))
    plon, plat = np.deg2rad(ports[["port_lon", "port_lat"]].to_numpy()).T
    seeds = np.column_stack((np.cos(plat) * np.cos(plon), np.cos(plat) * np.sin(plon), np.sin(plat)))
    distance, indices = cKDTree(sphere).query(seeds)
    selected = pd.concat([ports.reset_index(drop=True),
                          grid.iloc[indices].reset_index(drop=True)], axis=1)
    selected["port_to_grid_km"] = 6371 * 2 * np.arcsin(np.minimum(distance / 2, 1))
    if (selected.port_to_grid_km > 150).any():
        raise ValueError("A validation-port seed is over 150 km from a coastal site; review it.")
    return selected


def wave_number(period, depth):
    period, depth = np.broadcast_arrays(np.asarray(period, dtype=float), np.asarray(depth, dtype=float))
    if (np.any(~np.isfinite(period)) or np.any(~np.isfinite(depth))
            or np.any(period <= 0) or np.any(depth <= 0)):
        raise ValueError("Wave periods and depths must be finite and positive.")
    omega2 = (2 * np.pi / period) ** 2
    k = np.maximum(omega2 / GRAVITY, np.sqrt(omega2 / (GRAVITY * depth)))
    for _ in range(40):
        t = np.tanh(k * depth)
        residual = GRAVITY * k * t - omega2
        derivative = GRAVITY * (t + k * depth * (1 - t * t))
        update = residual / derivative
        k = k - update
        if np.all(np.abs(update) <= 1e-12 * np.maximum(k, 1)):
            break
    if not np.allclose(GRAVITY * k * np.tanh(k * depth), omega2, rtol=1e-9, atol=1e-12):
        raise ArithmeticError("Dispersion relation failed to converge.")
    return k


def group_speed(period, depth):
    k = wave_number(period, depth)
    kh2 = 2 * k * depth
    return 0.5 * (1 + kh2 / np.sinh(np.minimum(kh2, 700))) * (2 * np.pi / period) / k


def nearshore_waves(hs, period, direction, offshore_depth: float,
                    nearshore_depth: float, normal: float) -> dict[str, np.ndarray]:
    hs, period, direction = np.broadcast_arrays(np.asarray(hs, float), np.asarray(period, float),
                                                 np.asarray(direction, float))
    if (np.any(hs < 0) or np.any(~np.isfinite(hs)) or np.any(~np.isfinite(direction))
            or not np.isfinite([normal, offshore_depth, nearshore_depth]).all()
            or min(offshore_depth, nearshore_depth) <= 0):
        raise ValueError("Wave heights must be finite and nonnegative; normal must be finite.")
    depth0 = max(offshore_depth, nearshore_depth)
    k0, kn = wave_number(period, depth0), wave_number(period, nearshore_depth)
    c0, cn = (2 * np.pi / period) / k0, (2 * np.pi / period) / kn
    angle = np.deg2rad((direction - normal + 180) % 360 - 180)
    shoreward = np.cos(angle) > 0
    theta_near = np.arcsin(np.clip(np.sin(angle) * cn / c0, -1, 1))
    ks = np.sqrt(group_speed(period, depth0) / group_speed(period, nearshore_depth))
    kr = np.sqrt(np.maximum(np.cos(angle), 0) / np.cos(theta_near))
    kr = np.maximum(kr, 0.8)
    all_directions = np.minimum(hs * ks, 0.73 * nearshore_depth)
    short_waves = np.where(shoreward, np.minimum(hs * ks * kr, 0.73 * nearshore_depth), 0)
    return {"short_wave": short_waves, "all_direction_wave": all_directions,
            "ks": ks, "kr": kr, "shoreward": shoreward, "near_angle": theta_near}


def _log_cosh(value):
    return np.logaddexp(value, -value) - np.log(2)


def difference_kernel(f1, delta_f, delta_theta, depth: float):
    """Appendix A difference-interaction coefficient, transcribed from the PDF equation."""
    f2 = f1 + delta_f
    k1, k2 = wave_number(1 / f1, depth), wave_number(1 / f2, depth)
    cosine = np.cos(delta_theta)
    k3 = np.sqrt(np.maximum(k1 * k1 + k2 * k2 - 2 * k1 * k2 * cosine, 0))
    denominator = GRAVITY * k3 * np.tanh(k3 * depth) - (2 * np.pi * delta_f) ** 2
    if np.any(denominator <= 0):
        raise ArithmeticError("Nonpositive difference-wave interaction denominator.")
    ratio = np.exp(_log_cosh(k3 * depth) - _log_cosh(k1 * depth) - _log_cosh(k2 * depth))
    cosine_plus_pi = -cosine
    first = GRAVITY * k1 * k2 * cosine_plus_pi / (8 * np.pi**2 * f1 * f2) * ratio
    bracket = (
        delta_f * ((2 * np.pi)**4 * (f1 * f2)**2 / GRAVITY**2
                   - k1 * k2 * cosine_plus_pi)
        - 0.5 * (-f1 * k2**2 * np.exp(-2 * _log_cosh(k2 * depth))
                 + f2 * k1**2 * np.exp(-2 * _log_cosh(k1 * depth)))
    )
    return first - GRAVITY * delta_f / (denominator * f1 * f2) * bracket


def bound_ig_unit(period: float, depth: float = 20, gamma: float = 3.3,
                  n_frequency: int = 192, n_direction: int = 72,
                  n_ig: int = 19) -> tuple[float, float]:
    """Bound IG Hm0 and Tm01 for unit Hs; a documented numerical reconstruction.

    JONSWAP is normalized to m0=1/16. The directional density is proportional to
    cos(theta/2)**(2*s), s=7 for Tp>10 s, otherwise 2. Constant gamma=3.3 replaces
    the unpublished gamma(Hs,Tp) implementation. IG frequencies span .004-.04 Hz.
    """
    if period <= 0 or depth <= 0 or gamma < 1:
        raise ValueError("Tp and depth must be positive; gamma must be at least one.")
    if min(n_frequency, n_direction, n_ig) < 8:
        raise ValueError("Insufficient spectral quadrature resolution.")
    frequencies = np.geomspace(0.2 / period, 5 / period, n_frequency)
    fp = 1 / period

    def shape(f):
        sigma = np.where(f <= fp, 0.07, 0.09)
        peak = np.exp(-0.5 * ((f / fp - 1) / sigma)**2)
        return f**-5 * np.exp(-1.25 * (fp / f)**4) * gamma**peak

    norm = integrate.trapezoid(shape(frequencies), frequencies) * 16
    directions = np.linspace(-np.pi, np.pi, n_direction, endpoint=False)
    step = 2 * np.pi / n_direction
    spreading = np.maximum(np.cos(directions / 2), 0)**(2 * (7 if period > 10 else 2))
    spreading /= spreading.sum() * step
    lag_probability = np.array([np.sum(spreading * np.roll(spreading, j)) * step**2
                                for j in range(n_direction)])
    delta_theta = np.arange(n_direction) * step
    ig_frequencies = np.linspace(0.004, 0.04, n_ig)
    energy = []
    for delta in ig_frequencies:
        active = frequencies >= delta
        f = frequencies[active]
        coefficient = difference_kernel(f[:, None], delta, delta_theta[None, :], depth)
        directional_mean = coefficient**2 @ lag_probability
        integrand = directional_mean * shape(f) * shape(f + delta) / norm**2
        energy.append(2 * integrate.trapezoid(integrand, f))
    m0 = integrate.trapezoid(energy, ig_frequencies)
    m1 = integrate.trapezoid(np.asarray(energy) * ig_frequencies, ig_frequencies)
    if not np.isfinite(m0) or m0 <= 0 or m1 <= 0:
        raise ArithmeticError("Invalid bound-IG spectral moments.")
    return float(4 * np.sqrt(m0)), float(m0 / m1)


def ig_lookup(path: Path, depth: float, period_step: float = 0.1,
              n_frequency: int = 192, n_direction: int = 72, n_ig: int = 19,
              max_height: float = 12.5, max_period: float = 25.0) -> xr.Dataset:
    if (not np.isfinite([period_step, max_height, max_period]).all()
            or min(period_step, max_height) <= 0 or max_period < period_step):
        raise ValueError("Invalid IG lookup bounds or period step.")
    settings = {"depth_m": float(depth), "gamma": 3.3, "period_step": period_step,
                "n_frequency": n_frequency, "n_direction": n_direction, "n_ig": n_ig,
                "method": "paper-appendix-A-constant-gamma-v1"}
    if max_height != 12.5 or max_period != 25:
        settings.update(max_height=max_height, max_period=max_period)
    if path.exists():
        with xr.open_dataset(path) as source:
            ds = source.load()
        if ds.attrs["settings"] != json.dumps(settings, sort_keys=True):
            raise ValueError(f"IG cache settings differ: {path}")
        return ds
    periods = np.arange(period_step, max_period + period_step / 2, period_step)
    periods = np.unique(np.r_[periods[periods < max_period], max_period])
    unit = np.array([bound_ig_unit(tp, depth, n_frequency=n_frequency,
                                 n_direction=n_direction, n_ig=n_ig) for tp in periods])
    heights = np.arange(0, max_height + 0.025, 0.05)
    heights = np.unique(np.r_[heights[heights < max_height], max_height])
    ds = xr.Dataset(
        {"ig_height": (("hs", "tp"), heights[:, None]**2 * unit[None, :, 0]),
         "ig_period": (("hs", "tp"), np.broadcast_to(unit[None, :, 1], (len(heights), len(periods))))},
        coords={"hs": heights, "tp": periods},
        attrs={"settings": json.dumps(settings, sort_keys=True),
               "note": f"Reconstructed lookup, not the original authors' table; Hs=0..{max_height:g} inclusive."},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)
    return ds


def interpolate_ig(lookup: xr.Dataset, hs, peak_period) -> np.ndarray:
    hs, peak_period = np.broadcast_arrays(np.asarray(hs, float), np.asarray(peak_period, float))
    if (np.any(~np.isfinite(hs)) or np.any(~np.isfinite(peak_period))
            or np.any(hs < 0) or np.any(peak_period <= 0)):
        raise ValueError("Invalid wave input for IG interpolation.")
    outside = ((hs > float(lookup.hs.max())) | (peak_period < float(lookup.tp.min()))
               | (peak_period > float(lookup.tp.max())))
    if outside.any():
        raise ValueError(f"{outside.sum()} wave states exceed the IG lookup; extend it explicitly.")
    return lookup.ig_height.interp(
        hs=xr.DataArray(hs, dims="sample"),
        tp=xr.DataArray(peak_period, dims="sample"),
    ).to_numpy()


def merge_short_gaps(unavailable, gap_hours: int = 4) -> np.ndarray:
    result = np.asarray(unavailable, dtype=bool).copy()
    if result.ndim != 1 or gap_hours < 0:
        raise ValueError("Expected a one-dimensional hourly mask and nonnegative gap.")
    boundaries = np.flatnonzero(np.diff(np.r_[False, ~result, False]))
    for start, end in boundaries.reshape(-1, 2):
        if start > 0 and end < len(result) and end - start <= gap_hours:
            result[start:end] = True
    return result


def event_statistics(unavailable, times: pd.DatetimeIndex) -> dict[str, float]:
    unavailable = np.asarray(unavailable, bool)
    if not len(times) or len(times) != len(unavailable):
        raise ValueError("Nonempty matching time and event arrays required.")
    if len(times) > 1 and not (np.diff(times.asi8) == pd.Timedelta(hours=1).value).all():
        raise ValueError("Event statistics require contiguous hourly observations.")
    edges = np.flatnonzero(np.diff(np.r_[False, unavailable, False]))
    runs = edges.reshape(-1, 2)
    duration = runs[:, 1] - runs[:, 0]
    years = len(times) / (365.25 * 24)
    return {"availability": 100 * (1 - unavailable.mean()),
            "frequency": len(runs) / years, "duration": float(duration.mean()) if len(runs) else 0,
            "events": len(runs), "hours": len(times)}


def time_segments(index: pd.DatetimeIndex):
    if not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError("Time series must be sorted without duplicate timestamps.")
    breaks = np.flatnonzero(np.diff(index.asi8) != pd.Timedelta(hours=1).value) + 1
    return np.split(np.arange(len(index)), breaks)


def conditional_statistics(frame: pd.DataFrame, scenario: Scenario) -> dict[str, float]:
    if frame.empty or not np.isfinite(frame[["wind", "short_wave", "ig"]]).all().all():
        raise ValueError("Conditional statistics require nonempty, complete indicator data.")
    limits = {"wind": scenario.wind_limit, "short_wave": scenario.wave_limit,
              "ig": scenario.ig_limit}
    totals = {name: [] for name in [*limits, "total"]}
    raw_totals = []
    total_events = total_hours = total_duration = 0
    for positions in time_segments(frame.index):
        part = frame.iloc[positions]
        masks = {name: part[name].to_numpy() >= limit for name, limit in limits.items()}
        combined = np.logical_or.reduce(list(masks.values()))
        raw_totals.extend(combined.tolist())
        masks["total"] = combined
        for name, mask in masks.items():
            conditional = merge_short_gaps(mask, scenario.gap_hours)
            totals[name].extend(conditional.tolist())
            if name == "total":
                stats = event_statistics(conditional, part.index)
                total_events += stats["events"]
                total_duration += stats["events"] * stats["duration"]
                total_hours += stats["hours"]
    result = {f"{name}_availability": 100 * (1 - np.mean(mask)) for name, mask in totals.items()}
    result.update(frequency=total_events / (total_hours / (365.25 * 24)),
                  duration=total_duration / total_events if total_events else 0,
                  events=total_events,
                  raw_availability=100 * (1 - np.mean(raw_totals)),
                  observed_hours=total_hours)
    for name in limits:
        values = frame[name]
        if name == "short_wave" and "shoreward" in frame:
            values = frame.loc[frame.shoreward, name]
            if values.empty:
                LOGGER.warning("No shoreward waves in this subset; wave percentiles are undefined")
        result[f"{name}_p50"] = float(values.quantile(0.5))
        result[f"{name}_p95"] = float(values.quantile(0.95))
        result[f"{name}_raw_availability"] = 100 * (frame[name] < limits[name]).mean()
    return result


def prepare_site(ds: xr.Dataset, site: pd.Series, lookup: xr.Dataset,
                 scenario: Scenario) -> pd.DataFrame:
    if ds.longitude.ndim == 0 and ds.latitude.ndim == 0:
        selected = ds
    else:
        selected = ds.sel(longitude=float(site.lon), latitude=float(site.lat), method="nearest")
    if (abs(float(selected.longitude) - site.lon) > 1e-6
            or abs(float(selected.latitude) - site.lat) > 1e-6):
        raise ValueError("Downloaded data do not contain the selected ERA5 site.")
    required = ["u10", "v10", "swh", "mwd", "mwp", "pp1d"]
    missing = set(required) - set(selected.data_vars)
    if missing:
        raise ValueError(f"Missing ERA5 variables: {sorted(missing)}")
    raw = selected[required].to_dataframe()[required].sort_index()
    if not isinstance(raw.index, pd.DatetimeIndex):
        raise ValueError("Unexpected dimensions after ERA5 point selection.")
    valid = np.isfinite(raw).all(axis=1) & (raw.swh >= 0) & (raw.mwp > 0) & (raw.pp1d > 0)
    if not valid.all():
        LOGGER.warning("Site %s: excluding %d missing/invalid hours without bridging their gaps",
                       site.site_id, (~valid).sum())
    raw = raw.loc[valid]
    if raw.empty:
        raise ValueError(f"No valid data at site {site.site_id}.")
    waves = nearshore_waves(raw.swh.to_numpy() * scenario.wave_factor, raw.mwp.to_numpy(),
                            raw.mwd.to_numpy(), float(site.depth0), scenario.depth_m,
                            float(site.shore_normal))
    frame = pd.DataFrame({
        "wind": np.hypot(raw.u10, raw.v10) * scenario.wind_factor,
        "short_wave": waves["short_wave"],
        "ig": interpolate_ig(lookup, waves["all_direction_wave"], raw.pp1d.to_numpy()),
        "hs_offshore": raw.swh, "hs_all_direction": waves["all_direction_wave"],
        "mean_period": raw.mwp, "peak_period": raw.pp1d,
        "ks": waves["ks"], "kr": waves["kr"],
        "shoreward": waves["shoreward"],
        "sediment_proxy": waves["short_wave"]**2.5 * np.abs(np.sin(2 * waves["near_angle"])),
    }, index=raw.index)
    return frame


def risk_class(availability):
    return pd.cut(availability, [-np.inf, 75, 85, 95, np.inf],
                  labels=["Very high", "High", "Medium", "Low"], right=True)


def main_driver(row: pd.Series) -> str:
    wind, short, ig = [row[f"{name}_availability"] for name in ("wind", "short_wave", "ig")]
    if wind < min(short, ig):
        return "Wind"
    if abs(short - ig) < 1.0:
        return "Short & IG waves"
    return "Short waves" if short < ig else "IG waves"


def summarise_frame(frame: pd.DataFrame, site: pd.Series, scenario: Scenario):
    """Summarise already transformed observations, reusable for multiple time windows."""
    identity = {"site_id": int(site.site_id), "lon": float(site.lon), "lat": float(site.lat),
                "name": site.get("name", f"Site {int(site.site_id)}"), "scenario": scenario.name}
    overall = [{**identity, "depth0": float(site.depth0),
                "shore_normal": float(site.shore_normal), "depth_m": scenario.depth_m,
                "mean_period": float(frame.mean_period.mean()),
                "peak_period": float(frame.peak_period.mean()),
                "ks": float(frame.ks.mean()), "kr": float(frame.kr.mean()),
                "sediment_proxy": float(frame.sediment_proxy.mean()),
                **conditional_statistics(frame, scenario)}]
    yearly, seasonal = [], []
    for year, part in frame.groupby(frame.index.year):
        yearly.append({**identity, "year": int(year), **conditional_statistics(part, scenario)})
    labels = np.array(["DJF", "MAM", "JJA", "SON"])[(frame.index.month % 12) // 3]
    for season, part in frame.groupby(labels):
        statistics = conditional_statistics(part, scenario)
        statistics["frequency_per_season"] = statistics["events"] / part.index.year.nunique()
        seasonal.append({**identity, "season": season, **statistics})
    overall = pd.DataFrame(overall)
    overall["risk"] = risk_class(overall.total_availability)
    overall["main_driver"] = overall.apply(main_driver, axis=1)
    return overall, pd.DataFrame(yearly), pd.DataFrame(seasonal)


def analyse_sites(ds: xr.Dataset, sites: pd.DataFrame, scenario: Scenario,
                  lookup: xr.Dataset):
    overall, yearly, seasonal, series = [], [], [], {}
    for _, site in sites.iterrows():
        frame = prepare_site(ds, site, lookup, scenario)
        series[int(site.site_id)] = frame
        summary, annual, seasons = summarise_frame(frame, site, scenario)
        overall.append(summary)
        yearly.append(annual)
        seasonal.append(seasons)
    return (pd.concat(overall, ignore_index=True), pd.concat(yearly, ignore_index=True),
            pd.concat(seasonal, ignore_index=True), series)


def _point_cache(root: Path, lon: float, lat: float, year: int, months: list[int]) -> Path:
    month_tag = "-".join(f"{m:02d}" for m in sorted(months))
    return (data_directories(root)["processed"] / "hourly"
            / f"lon{lon:+08.3f}_lat{lat:+07.3f}" / f"{year}_{month_tag}.nc")


def cache_hourly_points(ds: xr.Dataset, sites: pd.DataFrame, year: int,
                       months: list[int], root: Path) -> None:
    expected = pd.date_range(f"{year}-01-01", f"{year + 1}-01-01", freq="h", inclusive="left")
    expected = expected[expected.month.isin(months)]
    for site in sites.itertuples(index=False):
        point = ds.sel(longitude=float(site.lon), latitude=float(site.lat))
        point = point.sel(time=expected)
        path = _point_cache(root, site.lon, site.lat, year, months)
        path.parent.mkdir(parents=True, exist_ok=True)
        encoding = {name: {"dtype": "float32", "zlib": True, "complevel": 4}
                    for name in point.data_vars if point[name].dtype.kind == "f"}
        point.to_netcdf(path, encoding=encoding)


def download_plan(sites: pd.DataFrame, years: list[int], months: list[int]) -> list[dict]:
    if not years or not months or any(m < 1 or m > 12 for m in months):
        raise ValueError("Nonempty years and valid calendar months required.")
    if sites.empty or sites.duplicated(["lon", "lat"]).any():
        raise ValueError("Sites must be nonempty and have unique coordinates.")
    grouped = sites.assign(
        tile_lon=np.floor(sites.lon / 5).astype(int) * 5,
        tile_lat=np.floor(sites.lat / 5).astype(int) * 5,
    )
    plans = []
    for year in years:
        for month in months:
            for (lon, lat), group in grouped.groupby(["tile_lon", "tile_lat"], sort=True):
                area = [min(90, int(lat) + 5), int(lon), int(lat), min(179.5, int(lon) + 5)]
                request = era5_request(year, [month], area)
                digest = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()[:12]
                plans.append({"year": year, "month": month, "request": request,
                              "site_ids": group.site_id.tolist(),
                              "filename": f"era5-tile-{year}-{month:02d}-{digest}.nc"})
    return plans


def download_hourly_sites(sites: pd.DataFrame, years: list[int], months: list[int],
                         root: Path = Path("data"), request_budget: int = 2,
                         allow_large_download: bool = False) -> None:
    plans = download_plan(sites, years, months)
    paths = data_directories(root)
    needed = [plan for plan in plans
              if any(not _point_cache(root, float(site.lon), float(site.lat),
                                      plan["year"], [plan["month"]]).exists()
                     for site in sites.loc[sites.site_id.isin(plan["site_ids"])].itertuples())]
    if len(needed) > request_budget and not allow_large_download:
        raise ValueError(f"{len(needed)} tiled requests exceed budget {request_budget}; "
                         "review the plan before explicitly authorizing global downloads.")
    import shutil
    estimated_bytes = 0
    for plan in needed:
        north, west, south, east = plan["request"]["area"]
        cells = (round((north - south) / 0.5) + 1) * (round((east - west) / 0.5) + 1)
        hours = calendar.monthrange(plan["year"], plan["month"])[1] * 24
        if not (paths["raw"] / plan["filename"]).exists():
            estimated_bytes += cells * hours * 6 * 4 * 2
        estimated_bytes += len(plan["site_ids"]) * hours * 6 * 4
    if shutil.disk_usage(paths["raw"]).free < max(estimated_bytes, 2 * 1024**3):
        raise OSError(f"Insufficient free space for the retained tile and point caches "
                      f"(conservative estimate {estimated_bytes / 1024**3:.1f} GiB).")
    for plan in needed:
        group = sites.loc[sites.site_id.isin(plan["site_ids"])]
        raw_path = paths["raw"] / plan["filename"]
        retrieve_era5(plan["request"], raw_path)
        cache_hourly_points(open_era5(raw_path), group, plan["year"], [plan["month"]], root)


def load_site_history(site: pd.Series, years: list[int], months: list[int],
                      root: Path = Path("data")) -> xr.Dataset:
    datasets = []
    for year in years:
        for month in months:
            path = _point_cache(root, site.lon, site.lat, year, [month])
            if not path.exists():
                raise FileNotFoundError(f"Missing point cache; run the download stage first: {path}")
            with xr.open_dataset(path) as source:
                datasets.append(source.load())
    return xr.concat(datasets, dim="time", data_vars="minimal", coords="minimal",
                     compat="override").sortby("time")


def run_study(sites: pd.DataFrame, years: list[int], months: list[int],
              root: Path = Path("data"), scenarios: Optional[list[Scenario]] = None):
    scenarios = SCENARIOS if scenarios is None else scenarios
    processed = data_directories(root)["processed"]
    lookups = {depth: ig_lookup(processed / f"ig-lookup-{depth:g}m.nc", depth)
               for depth in sorted({s.depth_m for s in scenarios})}
    overall, yearly, seasonal = [], [], []
    for _, site in sites.iterrows():
        ds = load_site_history(site, years, months, root)
        for scenario in scenarios:
            summary, annual, seasons, series = analyse_sites(
                ds, site.to_frame().T, scenario, lookups[scenario.depth_m])
            overall.append(summary)
            yearly.append(annual)
            seasonal.append(seasons)
            if scenario.name == "baseline":
                path = processed / f"indicators-site-{int(site.site_id)}.nc"
                xr.Dataset.from_dataframe(series[int(site.site_id)]).to_netcdf(path)
    results = tuple(pd.concat(parts, ignore_index=True) for parts in (overall, yearly, seasonal))
    for name, result in zip(("site-summary", "annual", "seasonal"), results):
        result.to_csv(processed / f"{name}.csv", index=False)
    baseline = results[0].loc[results[0].scenario == "baseline"]
    legacy_names = {
        "shore_normal": "shoreN", "mean_period": "tm", "peak_period": "tp",
        "ks": "Ks", "kr": "Kr", "wind_p50": "wind50", "wind_p95": "wind95",
        "wind_raw_availability": "windAv", "short_wave_p50": "swh50",
        "short_wave_p95": "swh95", "short_wave_raw_availability": "swhAv",
        "ig_p50": "ig50", "ig_p95": "ig95", "ig_raw_availability": "igAv",
        "wind_availability": "windAvUp", "short_wave_availability": "swhAvUp",
        "ig_availability": "igAvUp", "total_availability": "operability",
        "sediment_proxy": "wavePo",
    }
    baseline.rename(columns=legacy_names).to_csv(processed / "legacy-compatible-summary.csv", index=False)
    export_points(baseline, processed / "operability.geojson",
                  "Calculated ERA5 screening indicators; reconstructed method, not measured port downtime.")
    return results


def export_points(frame: pd.DataFrame, path: Path, description: str):
    features = []
    for row in frame.to_dict(orient="records"):
        lon, lat = float(row.pop("lon")), float(row.pop("lat"))
        properties = {}
        for name, value in row.items():
            properties[name] = None if pd.isna(value) else value
        features.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [lon, lat]},
                         "properties": properties})
    write_json(path, {"type": "FeatureCollection", "description": description, "features": features})


def run_manifest(root: Path, settings: dict) -> Path:
    import importlib.metadata
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "settings": settings,
                "reference": "https://doi.org/10.3390/jmse9070695",
                "packages": {package: importlib.metadata.version(package)
                             for package in ["numpy", "pandas", "scipy", "xarray", "cdsapi"]},
                "download_manifests": [], "reference_manifests": []}
    for path in sorted(Path(root).rglob("*.request.json")):
        manifest["download_manifests"].append(
            {"file": str(path.relative_to(root)),
             "cached_file_exists": Path(str(path)[:-len(".request.json")]).is_file(),
             **json.loads(path.read_text())})
    for path in sorted(Path(root).rglob("*.source.json")):
        manifest["reference_manifests"].append(
            {"file": str(path.relative_to(root)), **json.loads(path.read_text())})
    path = data_directories(root)["processed"] / "run-manifest.json"
    write_json(path, manifest)
    return path
