"""Continent-sized acquisition and resumable analysis for the full-method notebook."""

from __future__ import annotations

import calendar
import hashlib
import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely import STRtree
from shapely.affinity import translate

import port_operability as model

CONTINENTS = ("Africa", "Asia", "Europe", "North America", "Oceania", "South America")
VARIABLES = ["u10", "v10", "swh", "mwd", "mwp", "pp1d"]
PERIODS = {"paper_1979_2018": ("1979-01-01", "2019-01-01"),
           "extended_1979_2025": ("1979-01-01", "2026-01-01")}


class CDSWaitTimeout(TimeoutError):
    """The submitted CDS job did not finish within its polling budget."""


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def grid_digest(sites: pd.DataFrame) -> str:
    columns = ["site_id", "lon", "lat", "depth0", "shore_normal"]
    return _digest(sites.sort_values("site_id")[columns].to_dict("records"))


def assign_continents(sites: gpd.GeoDataFrame, countries: gpd.GeoDataFrame) -> pd.DataFrame:
    """Use nearest country to the coast anchor; groups follow Natural Earth CONTINENT."""
    frame = pd.DataFrame(sites.drop(columns="geometry")).copy()
    frame["lon"], frame["lat"] = sites.geometry.x, sites.geometry.y
    if sites.crs is None or sites.crs.to_epsg() != 4326:
        raise ValueError("The saved coastal grid must use EPSG:4326.")
    if frame.site_id.duplicated().any() or frame.duplicated(["lon", "lat"]).any():
        raise ValueError("The saved grid contains duplicate site identifiers or coordinates.")
    shapes = [translate(geometry, xoff=shift)
              for shift in (-360, 0, 360) for geometry in countries.geometry]
    points = gpd.points_from_xy(frame.anchor_lon, frame.anchor_lat)
    queries, matches = STRtree(shapes).query_nearest(points, all_matches=False)
    country_index = np.empty(len(frame), dtype=int)
    country_index[queries] = matches % len(countries)
    frame["country"] = countries.iloc[country_index].ADMIN.to_numpy()
    frame["continent"] = countries.iloc[country_index].CONTINENT.to_numpy()
    if not frame.continent.isin(CONTINENTS).all():
        invalid = frame.loc[~frame.continent.isin(CONTINENTS), ["site_id", "country", "continent"]]
        raise ValueError(f"Review unsupported continent assignments:\n{invalid}")
    return frame


def _months(years, months):
    years, months = list(years), list(months)
    if (not years or not months or len(set(years)) != len(years)
            or len(set(months)) != len(months) or any(m not in range(1, 13) for m in months)):
        raise ValueError("Unique nonempty years and valid calendar months required.")
    return [(int(y), int(m)) for y in sorted(years) for m in sorted(months)]


def _group(sites, continents):
    if not continents or len(set(continents)) != len(continents):
        raise ValueError("Select one or more distinct continents.")
    if not set(continents) <= set(CONTINENTS):
        raise ValueError(f"Supported continent groups: {CONTINENTS}")
    for continent in continents:
        group = sites.loc[sites.continent.eq(continent)].sort_values("site_id").copy()
        if group.empty:
            raise ValueError(f"No selected sites in {continent}.")
        yield continent, group


def month_path(root: Path, continent: str, year: int, month: int) -> Path:
    return Path(root) / "monthly" / continent.lower().replace(" ", "-") / f"{year}-{month:02d}.nc"


def _times(year: int, month: int) -> pd.DatetimeIndex:
    return pd.date_range(f"{year}-{month:02d}-01", periods=calendar.monthrange(year, month)[1] * 24,
                         freq="h")


def download_estimate(sites, continents, years, root: Path, months=range(1, 13)) -> pd.DataFrame:
    """Conservative uncompressed sizes, before raw-response reuse or NetCDF compression."""
    pairs = _months(years, months)
    rows = []
    for continent, group in _group(sites, continents):
        template = model.download_plan(group, [pairs[0][0]], [pairs[0][1]])
        cells = 0
        for item in template:
            north, west, south, east = item["request"]["area"]
            cells += (round((north - south) / .5) + 1) * (round((east - west) / .5) + 1)
        pending = [(y, m) for y, m in pairs if not month_path(root, continent, y, m).exists()]
        hours = sum(calendar.monthrange(y, m)[1] * 24 for y, m in pending)
        peak_hours = max((calendar.monthrange(y, m)[1] * 24 for y, m in pending), default=0)
        rows.append({
            "continent": continent, "sites": len(group), "months": len(pairs),
            "cached_months": len(pairs) - len(pending),
            "pending_requests_upper_bound": len(template) * len(pending),
            "point_cache_gib": len(group) * hours * 6 * 4 / 1024**3,
            "raw_and_extraction_gib": cells * hours * 6 * 4 * 2 / 1024**3,
            "monthly_working_gib": cells * peak_hours * 6 * 4 * 2 / 1024**3,
        })
    return pd.DataFrame(rows)


def _check_month(ds: xr.Dataset, group: pd.DataFrame, year: int, month: int):
    if ds.attrs.get("grid_digest") != grid_digest(group):
        raise ValueError("Monthly cache grid differs; do not mix coastal-grid versions.")
    if not set(VARIABLES) <= set(ds.data_vars) or set(ds.sizes) != {"time", "site"}:
        raise ValueError("Monthly cache does not contain the six expected time/site variables.")
    if not np.array_equal(ds.site.values, group.site_id.to_numpy()):
        raise ValueError("Monthly cache site identifiers differ.")
    if not np.array_equal(ds.time.values, _times(year, month).values):
        raise ValueError("Monthly cache is missing or duplicates requested hourly timestamps.")
    if not (np.array_equal(ds.longitude.values, group.lon.to_numpy())
            and np.array_equal(ds.latitude.values, group.lat.to_numpy())):
        raise ValueError("Monthly cache coordinates differ.")


def _discard_new_response(path: Path):
    contents = path.with_suffix(path.suffix + ".contents")
    if contents.exists():
        members = list(contents.iterdir())
        if any(not p.is_file() or p.suffix != ".nc" for p in members):
            raise ValueError(f"Refusing cleanup of unexpected response contents: {contents}")
        for member in members:
            member.unlink()
        contents.rmdir()
    path.unlink()
    manifest = path.with_suffix(path.suffix + ".request.json")
    metadata = json.loads(manifest.read_text())
    metadata["discarded_after_verified_month_cache"] = True
    model.write_json(manifest, metadata)


def _save_month(ds, group, year, month, target, manifests):
    ds.attrs = {"grid_digest": grid_digest(group),
                "request_manifests": json.dumps(manifests)}
    _check_month(ds, group, year, month)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".nc.part")
    encoding = {v: {"dtype": "float32", "zlib": True, "complevel": 4,
                    "chunksizes": (len(_times(year, month)), min(8, len(group)))}
                for v in VARIABLES}
    ds.to_netcdf(partial, engine="netcdf4", encoding=encoding)
    with xr.open_dataset(partial) as verified:
        _check_month(verified, group, year, month)
    partial.replace(target)
    model.write_json(target.with_suffix(".nc.source.json"), {
        "sha256": model.file_hash(target), "grid_digest": grid_digest(group),
        "request_manifests": manifests,
    })


def download_continents(sites, continents, years, data_root: Path, root: Path,
                        request_budget: int, discard_new_raw: bool = False,
                        months=range(1, 13)):
    """Persist one paired-point NetCDF per continent/month; never download implicitly."""
    pairs = _months(years, months)
    report = download_estimate(sites, continents, years, root, months)
    requests = int(report.pending_requests_upper_bound.sum())
    if request_budget < 0 or requests > request_budget:
        raise ValueError(f"Up to {requests:,} requests exceed budget {request_budget:,}. Review the plan.")
    required = report.point_cache_gib.sum() + 2
    required += (report.monthly_working_gib.max() if discard_new_raw
                 else report.raw_and_extraction_gib.sum())
    if shutil.disk_usage(data_root).free / 1024**3 < required:
        raise OSError(f"Insufficient disk space: conservative incremental requirement {required:.1f} GiB.")
    raw_folder = model.data_directories(data_root)["raw"]
    for continent, group in _group(sites, continents):
        for year, month in pairs:
            target = month_path(root, continent, year, month)
            if target.exists():
                with xr.open_dataset(target) as ds:
                    _check_month(ds, group, year, month)
                continue
            print(f"{continent}: {year}-{month:02d}, {len(group)} sites", flush=True)
            parts, new_responses, requests_used = [], [], []
            for item in model.download_plan(group, [year], [month]):
                path = raw_folder / item["filename"]
                existed = path.exists()
                if not existed and path.with_suffix(path.suffix + ".contents").exists():
                    raise FileExistsError(f"Review an existing extraction folder before retrieval: {path}")
                model.retrieve_era5(item["request"], path)
                raw = model.open_era5(path)
                subset = group.loc[group.site_id.isin(item["site_ids"])]
                point = raw[VARIABLES].sel(
                    longitude=xr.DataArray(subset.lon.to_numpy(), dims="site"),
                    latitude=xr.DataArray(subset.lat.to_numpy(), dims="site"),
                    time=_times(year, month),
                ).assign_coords(site=subset.site_id.to_numpy()).transpose("time", "site")
                parts.append(point)
                requests_used.append(str(path.with_suffix(path.suffix + ".request.json")))
                if not existed:
                    new_responses.append(path)
            combined = xr.concat(parts, dim="site", data_vars="all", coords="minimal",
                                 compat="no_conflicts", join="exact").sortby("site")
            _save_month(combined, group, year, month, target, requests_used)
            if discard_new_raw:
                for path in new_responses:
                    _discard_new_response(path)


def _grib_messages(chunks):
    buffer = bytearray()
    for chunk in chunks:
        buffer.extend(chunk)
        offset = 0
        while len(buffer) - offset >= 8:
            if buffer[offset:offset + 4] != b"GRIB":
                raise ValueError("Expected a GRIB response, not an archive or error document.")
            edition = buffer[offset + 7]
            if edition == 1:
                length = int.from_bytes(buffer[offset + 4:offset + 7], "big")
            elif edition == 2:
                if len(buffer) - offset < 16:
                    break
                length = int.from_bytes(buffer[offset + 8:offset + 16], "big")
            else:
                raise ValueError(f"Unsupported GRIB edition: {edition}")
            if not 12 <= length <= 32 * 1024**2:
                raise ValueError(f"Invalid or oversized GRIB field: {length} bytes.")
            if len(buffer) - offset < length:
                break
            message = bytes(buffer[offset:offset + length])
            if message[-4:] != b"7777":
                raise ValueError("Invalid GRIB field terminator.")
            offset += length
            yield message
        del buffer[:offset]
    if buffer:
        raise ValueError("Truncated GRIB response.")


def _stream_grib_points(chunks, group, times):
    import eccodes

    parameters = {165: "u10", 166: "v10", 140229: "swh",
                  140230: "mwd", 140232: "mwp", 140231: "pp1d"}
    arrays = {v: np.full((len(times), len(group)), np.nan, dtype="float32") for v in VARIABLES}
    seen = {v: np.zeros(len(times), dtype=bool) for v in VARIABLES}
    time_index = {timestamp: i for i, timestamp in enumerate(times)}
    indices = {}
    digest, size = hashlib.sha256(), 0

    def checked_chunks():
        nonlocal size
        for chunk in chunks:
            digest.update(chunk)
            size += len(chunk)
            yield chunk

    for message in _grib_messages(checked_chunks()):
        handle = eccodes.codes_new_from_message(message)
        try:
            parameter = eccodes.codes_get(handle, "paramId")
            if parameter not in parameters:
                raise ValueError(f"Unexpected GRIB parameter: {parameter}")
            variable = parameters[parameter]
            date = eccodes.codes_get(handle, "validityDate")
            clock = eccodes.codes_get(handle, "validityTime")
            timestamp = pd.to_datetime(f"{date:08d}{clock:04d}", format="%Y%m%d%H%M")
            if timestamp not in time_index:
                raise ValueError(f"Unexpected GRIB timestamp: {timestamp}")
            row = time_index[timestamp]
            if seen[variable][row]:
                raise ValueError(f"Duplicate GRIB field: {variable} {timestamp}")
            grid_hash = eccodes.codes_get(handle, "md5GridSection")
            if grid_hash not in indices:
                longitude = (eccodes.codes_get_array(handle, "longitudes") + 180) % 360 - 180
                latitude = eccodes.codes_get_array(handle, "latitudes")
                cells = {(float(lon), float(lat)): i
                         for i, (lon, lat) in enumerate(zip(longitude, latitude))}
                coordinates = list(zip(group.lon, group.lat))
                if any(point not in cells for point in coordinates):
                    raise ValueError("Selected positions are not exact GRIB grid nodes.")
                indices[grid_hash] = np.array([cells[point] for point in coordinates])
            values = eccodes.codes_get_values(handle)[indices[grid_hash]]
            values[values == eccodes.codes_get(handle, "missingValue")] = np.nan
            arrays[variable][row] = values
            seen[variable][row] = True
        finally:
            eccodes.codes_release(handle)
    missing = {v: int((~complete).sum()) for v, complete in seen.items() if not complete.all()}
    if missing:
        raise ValueError(f"Missing GRIB hourly fields: {missing}")
    result = xr.Dataset(
        {v: (("time", "site"), values) for v, values in arrays.items()},
        coords={"time": times, "site": group.site_id.to_numpy(),
                "longitude": ("site", group.lon.to_numpy()),
                "latitude": ("site", group.lat.to_numpy())})
    return result, digest.hexdigest(), size


def _download_grib_file(remote, directory: Path, attempts: int = 3) -> Path:
    """Retain verified HTTP range prefixes so interrupted transfers can resume."""
    import requests

    if attempts < 1:
        raise ValueError("A positive transfer attempt limit is required.")
    result = remote.get_results()
    expected = int(result.content_length)
    if expected <= 0:
        raise ValueError("The prepared GRIB response must have a positive size.")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    identity = _digest({"request_id": remote.request_id, "bytes": expected})
    target = directory / f"{identity}.grib.part"
    offset = target.stat().st_size if target.exists() else 0
    if offset > expected:
        raise ValueError(f"Local GRIB prefix exceeds the prepared response size: {target}")
    if shutil.disk_usage(directory).free < expected - offset + 2 * 1024**3:
        raise OSError("Insufficient local disk space for the temporary GRIB and safety reserve.")
    interruptions = 0
    while offset < expected:
        end = min(expected - 1, offset + 64 * 1024**2 - 1)
        try:
            result = remote.get_results()
            if int(result.content_length) != expected:
                raise ValueError("Prepared GRIB response size changed during transfer.")
            with requests.get(
                    result.location, stream=True, timeout=(30, 120),
                    headers={"Range": f"bytes={offset}-{end}", "Accept-Encoding": "identity"}) as response:
                response.raise_for_status()
                if (response.status_code != 206
                        or response.headers.get("Content-Range") != f"bytes {offset}-{end}/{expected}"
                        or response.headers.get("Content-Encoding", "identity") != "identity"):
                    raise ValueError("Server did not honor the exact GRIB byte range; prefix retained.")
                with target.open("ab") as output:
                    for block in response.iter_content(chunk_size=1024**2):
                        if output.tell() + len(block) > end + 1:
                            raise ValueError("GRIB range body exceeds its declared bounds.")
                        output.write(block)
                if target.stat().st_size != end + 1:
                    raise requests.ConnectionError("Incomplete GRIB byte range.")
        except requests.RequestException as error:
            offset = target.stat().st_size if target.exists() else 0
            interruptions += 1
            print(f"GRIB transfer interrupted ({type(error).__name__}); retained "
                  f"{offset:,}/{expected:,} bytes.", flush=True)
            if interruptions >= attempts:
                raise RuntimeError(
                    f"GRIB transfer interrupted {interruptions} times; {offset:,}/{expected:,} "
                    f"bytes retained locally for job {remote.request_id}.") from None
            time.sleep(2 ** interruptions)
            continue
        offset = target.stat().st_size
        print(f"GRIB transfer: {offset:,}/{expected:,} bytes "
              f"({100 * offset / expected:.1f}%).", flush=True)
    return target


def download_grib_years(sites, continents, years, root: Path, request_budget: int,
                        max_wait_seconds: int = 1800, transfer_directory: Path | None = None):
    """Stream one regional annual GRIB job into checked monthly point caches."""
    import cdsapi
    import requests

    years = sorted({year for year, _ in _months(years, range(1, 13))})
    groups = list(_group(sites, continents))
    pending = []
    for continent, group in groups:
        for year in years:
            paths = [month_path(root, continent, year, month) for month in range(1, 13)]
            if all(p.exists() for p in paths):
                for month, path in enumerate(paths, 1):
                    with xr.open_dataset(path) as ds:
                        _check_month(ds, group, year, month)
                _monthly_input_digest(root, continent, [year], range(1, 13), group)
                continue
            manifest = Path(root) / "requests" / f"{continent.lower().replace(' ', '-')}-{year}.json"
            pending.append((continent, group, year, manifest))
    new_jobs = sum(not manifest.exists() for _, _, _, manifest in pending)
    if request_budget < 0 or new_jobs > request_budget:
        raise ValueError(f"Up to {new_jobs} new GRIB jobs exceed budget {request_budget}.")
    if max_wait_seconds <= 0:
        raise ValueError("GRIB job wait limit must be positive.")
    if not pending:
        print("All requested annual GRIB point caches are verified; no network retrieval.", flush=True)
        return
    Path(root).mkdir(parents=True, exist_ok=True)
    point_bytes = sum(len(group) * len(pd.date_range(
        f"{year}-01-01", f"{year + 1}-01-01", freq="h", inclusive="left")) * 6 * 4
        for _, group, year, _ in pending)
    if shutil.disk_usage(root).free < point_bytes + 2 * 1024**3:
        raise OSError("Insufficient disk space for annual GRIB point caches and safety reserve.")
    client = cdsapi.Client(quiet=True, debug=False, progress=False, retry_max=1, timeout=60)
    for continent, group, year, manifest in pending:
        area = [float(group.lat.max()), float(group.lon.min()),
                float(group.lat.min()), float(group.lon.max())]
        request = model.era5_request(year, list(range(1, 13)), area)
        request["data_format"] = "grib"
        identity = {"dataset": model.ERA5_DATASET, "request": request,
                    "grid_digest": grid_digest(group)}
        if manifest.exists():
            metadata = json.loads(manifest.read_text())
            if any(metadata.get(key) != value for key, value in identity.items()):
                raise ValueError(f"Annual GRIB request/grid differs: {manifest}")
            remote = client.client.get_remote(metadata["request_id"])
        else:
            costs = client.client.estimate_costs(model.ERA5_DATASET, request)
            if costs["cost"] > costs["limit"]:
                raise ValueError(f"Annual GRIB job exceeds service limits: {costs}")
            remote = client.client.submit(model.ERA5_DATASET, request)
            metadata = {**identity, "request_id": remote.request_id, "costs": costs}
            model.write_json(manifest, metadata)
        started = time.monotonic()
        last_status = None
        while True:
            status = remote.status
            if status != last_status:
                print(f"{continent} {year}: GRIB job {remote.request_id} {status}", flush=True)
                metadata["status"] = status
                model.write_json(manifest, metadata)
                last_status = status
            if status == "successful":
                break
            if status not in ("accepted", "running"):
                raise RuntimeError(f"Annual GRIB job {remote.request_id} ended with status {status}.")
            if time.monotonic() - started >= max_wait_seconds:
                raise CDSWaitTimeout(f"Job {remote.request_id} still {status}; saved for resume, not resubmitted.")
            time.sleep(10)
        result = remote.get_results()
        times = pd.date_range(f"{year}-01-01", f"{year + 1}-01-01", freq="h", inclusive="left")
        local_grib = None
        if transfer_directory is not None:
            local_grib = _download_grib_file(remote, transfer_directory)
            with local_grib.open("rb") as source:
                ds, sha256, size = _stream_grib_points(
                    iter(lambda: source.read(1024**2), b""), group, times)
        else:
            try:
                with requests.get(result.location, stream=True, timeout=(30, 120)) as response:
                    if not response.ok:
                        raise RuntimeError(f"GRIB transfer failed: HTTP {response.status_code}")
                    ds, sha256, size = _stream_grib_points(
                        response.iter_content(chunk_size=1024**2), group, times)
            except requests.RequestException as error:
                raise RuntimeError(f"GRIB network transfer failed: {type(error).__name__}; resume job {remote.request_id}.") from None
        if size != result.content_length:
            raise ValueError(f"Incomplete GRIB transfer: {size} of {result.content_length} bytes.")
        metadata.update(raw_sha256=sha256, raw_bytes=size, raw_saved=False,
                        retrieval_seconds=time.monotonic() - started)
        model.write_json(manifest, metadata)
        for month in range(1, 13):
            target = month_path(root, continent, year, month)
            _save_month(ds.sel(time=_times(year, month)), group, year, month, target, [str(manifest)])
        _monthly_input_digest(root, continent, [year], range(1, 13), group)
        metadata["point_caches_verified"] = True
        model.write_json(manifest, metadata)
        if local_grib is not None:
            local_grib.unlink()
        print(f"Verified {year}: 12 monthly caches, {len(group)} positions, {len(times)} hours.", flush=True)


def _load_batch(group, continent, years, months, root):
    parts = []
    for year, month in _months(years, months):
        path = month_path(root, continent, year, month)
        if not path.exists():
            raise FileNotFoundError(f"Download missing monthly cache first: {path}")
        with xr.open_dataset(path) as ds:
            parts.append(ds.sel(site=group.site_id.to_numpy()).load())
    return xr.concat(parts, dim="time", data_vars="minimal", coords="minimal",
                     compat="equals", join="exact")


def _site_complete(folder: Path, expected: dict) -> bool:
    marker = folder / "complete.json"
    if not marker.exists():
        return False
    if json.loads(marker.read_text()) != expected:
        raise ValueError(f"Checkpoint settings differ: {folder}")
    if not all((folder / f"{name}.csv").exists() for name in ("summary", "annual", "seasonal")):
        raise FileNotFoundError(f"A completed checkpoint is missing result files: {folder}")
    return True


def _monthly_input_digest(root, continent, years, months, group):
    sources = []
    for year, month in _months(years, months):
        path = month_path(root, continent, year, month)
        metadata = json.loads(path.with_suffix(".nc.source.json").read_text())
        if metadata["grid_digest"] != grid_digest(group):
            raise ValueError(f"Monthly source grid differs: {path}")
        if model.file_hash(path) != metadata["sha256"]:
            raise ValueError(f"Monthly cache hash differs from its source manifest: {path}")
        sources.append((year, month, metadata["sha256"]))
    return _digest(sources)


def analyse_continents(sites, continents, years, root: Path, scenarios=None, periods=None,
                       batch_size: int = 8, save_hourly: bool = False,
                       max_ig_period: float = 40, months=range(1, 13)):
    """Transform once/site/scenario, summarise each study period, checkpoint each site."""
    if batch_size < 1:
        raise ValueError("Analysis batch size must be positive.")
    scenarios = model.SCENARIOS if scenarios is None else scenarios
    periods = PERIODS if periods is None else periods
    if not scenarios or not periods:
        raise ValueError("At least one scenario and study period required.")
    settings = {
        "years": list(years), "months": list(months), "periods": periods,
        "scenarios": [asdict(s) for s in scenarios], "max_ig_period": max_ig_period,
        "save_hourly": save_hourly,
        "source_sha256": {p.name: model.file_hash(p)
                          for p in [Path(model.__file__), Path(__file__)]},
    }
    signature = _digest(settings)
    lookups = {}
    collected = {name: [] for name in ("summary", "annual", "seasonal")}
    for continent, region in _group(sites, continents):
        region_root = Path(root) / "results" / continent.lower().replace(" ", "-")
        expected = {"signature": signature, "grid_digest": grid_digest(region),
                    "monthly_inputs": _monthly_input_digest(root, continent, years, months, region)}
        manifest = region_root / "settings.json"
        if manifest.exists() and json.loads(manifest.read_text()) != expected:
            raise ValueError(f"Analysis settings/grid changed; use a new results directory: {region_root}")
        model.write_json(manifest, expected)
        complete_sites = []
        validated = False
        for start in range(0, len(region), batch_size):
            batch = region.iloc[start:start + batch_size]
            pending = batch.loc[[
                not _site_complete(region_root / "sites" / str(int(s)), expected)
                for s in batch.site_id]]
            if not pending.empty:
                if not validated:
                    for year, month in _months(years, months):
                        with xr.open_dataset(month_path(root, continent, year, month)) as cache:
                            _check_month(cache, region, year, month)
                    validated = True
                if not lookups:
                    lookups = {depth: model.ig_lookup(
                        Path(root) / "lookups" / f"ig-{depth:g}m.nc", depth,
                        max_height=max(.73 * s.depth_m for s in scenarios), max_period=max_ig_period)
                        for depth in sorted({s.depth_m for s in scenarios})}
                ds = _load_batch(pending, continent, years, months, root)
                for _, site in pending.iterrows():
                    site_root = region_root / "sites" / str(int(site.site_id))
                    frames = {name: [] for name in collected}
                    point = ds.sel(site=int(site.site_id))
                    for scenario in scenarios:
                        frame = model.prepare_site(point, site, lookups[scenario.depth_m], scenario)
                        if save_hourly and scenario.name == "baseline":
                            site_root.mkdir(parents=True, exist_ok=True)
                            hourly = xr.Dataset.from_dataframe(frame)
                            hourly.to_netcdf(site_root / "hourly.nc", encoding={
                                v: {"zlib": True, "complevel": 4} for v in hourly.data_vars})
                        for period, (first, stop) in periods.items():
                            selected = frame.loc[(frame.index >= first) & (frame.index < stop)]
                            summary, annual, seasonal = model.summarise_frame(selected, site, scenario)
                            expected_hours = int((pd.Timestamp(stop) - pd.Timestamp(first)).total_seconds() / 3600)
                            summary["expected_hours"] = expected_hours
                            summary["coverage_fraction"] = summary.observed_hours / expected_hours
                            annual["expected_hours"] = annual.year.map(
                                lambda y: (366 if calendar.isleap(y) else 365) * 24)
                            annual["complete_year"] = annual.observed_hours.eq(annual.expected_hours)
                            for name, result in zip(frames, (summary, annual, seasonal)):
                                frames[name].append(result.assign(
                                    period=period, continent=continent, country=site.country))
                    site_root.mkdir(parents=True, exist_ok=True)
                    for name, parts in frames.items():
                        temporary = site_root / f"{name}.csv.part"
                        pd.concat(parts, ignore_index=True).to_csv(temporary, index=False)
                        temporary.replace(site_root / f"{name}.csv")
                    model.write_json(site_root / "complete.json", expected)
                    print(f"Analysed {continent} site {int(site.site_id)}", flush=True)
            complete_sites.extend(batch.site_id.tolist())
        for name in collected:
            result = pd.concat([pd.read_csv(region_root / "sites" / str(int(s)) / f"{name}.csv")
                                for s in complete_sites], ignore_index=True)
            result.to_csv(region_root / f"{name}.csv", index=False)
            collected[name].append(result)
    return tuple(pd.concat(parts, ignore_index=True) for parts in collected.values())
