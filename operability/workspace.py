"""Frozen inputs, local progress database and verifiable point archives."""

from __future__ import annotations

import calendar
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import json
import logging
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import uuid
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

import full_study as full
import port_operability as model
from .products import (
    atomic_csv, atomic_gzip_json, atomic_json, expected_product, export_tables,
    file_stamp, now, product_path, read_archive, verified_product,
)

LOGGER = logging.getLogger("operability")
DEFAULT_WORKSPACE = Path("operability-workspace")
SCHEMA = """
CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS sites (
    site_id INTEGER PRIMARY KEY, lon REAL NOT NULL, lat REAL NOT NULL,
    country TEXT NOT NULL, continent TEXT NOT NULL, region TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS jobs (
    region TEXT NOT NULL, year INTEGER NOT NULL, continent TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending', request_id TEXT, error TEXT,
    updated_at TEXT, PRIMARY KEY (region, year));
CREATE TABLE IF NOT EXISTS caches (
    region TEXT NOT NULL, year INTEGER NOT NULL, month INTEGER NOT NULL,
    path TEXT NOT NULL, sha256 TEXT NOT NULL, stamp TEXT NOT NULL,
    PRIMARY KEY (region, year, month),
    FOREIGN KEY (region, year) REFERENCES jobs(region, year));
CREATE TABLE IF NOT EXISTS point_months (
    site_id INTEGER NOT NULL REFERENCES sites(site_id),
    region TEXT NOT NULL, year INTEGER NOT NULL, month INTEGER NOT NULL,
    expected_hours INTEGER NOT NULL, valid_hours INTEGER NOT NULL,
    variable_hours TEXT NOT NULL, PRIMARY KEY (site_id, year, month),
    FOREIGN KEY (region, year, month) REFERENCES caches(region, year, month)
        ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS analyses (
    site_id INTEGER NOT NULL REFERENCES sites(site_id), period TEXT NOT NULL,
    status TEXT NOT NULL, method TEXT NOT NULL, inputs TEXT NOT NULL,
    stamp TEXT, error TEXT, updated_at TEXT NOT NULL,
    PRIMARY KEY (site_id, period));
CREATE INDEX IF NOT EXISTS point_months_region_year ON point_months(region, year);
CREATE INDEX IF NOT EXISTS analyses_status ON analyses(status, period);
"""


def periods_for(first: int, last: int) -> dict:
    if not 1940 <= first <= last < datetime.now(timezone.utc).year:
        raise ValueError("Select complete ERA5 calendar years, from 1940 onwards.")
    periods = {f"year_{year}": [f"{year}-01-01", f"{year + 1}-01-01"]
               for year in range(first, last + 1)}
    if first != last:
        name = "paper_1979_2018" if (first, last) == (1979, 2018) else f"extended_{first}_{last}"
        periods[name] = [f"{first}-01-01", f"{last + 1}-01-01"]
    if first <= 1979 and last >= 2018 and (first, last) != (1979, 2018):
        periods["paper_1979_2018"] = ["1979-01-01", "2019-01-01"]
    return periods

def period_years(bounds) -> list[int]:
    return list(range(pd.Timestamp(bounds[0]).year, pd.Timestamp(bounds[1]).year))

def assign_regions(sites: pd.DataFrame) -> pd.DataFrame:
    """Split only overly wide longitude spans, avoiding dateline-sized rectangles."""
    sites = sites.copy()
    sites["region"] = ""
    for continent, group in full._group(sites, sorted(sites.continent.unique())):
        pending, groups = [group], []
        while pending:
            part = pending.pop(0).sort_values(["lon", "site_id"])
            if part.lon.max() - part.lon.min() <= 180:
                groups.append(part)
            else:
                cut = int(np.argmax(np.diff(part.lon.to_numpy()))) + 1
                pending.extend([part.iloc[:cut], part.iloc[cut:]])
        for index, part in enumerate(sorted(groups, key=lambda g: g.lon.min()), 1):
            region = f"{continent.lower().replace(' ', '-')}-{index:02d}"
            sites.loc[part.index, "region"] = region
    return sites.sort_values("site_id")

def default_database(workspace_id: str) -> Path:
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_STATE_HOME", str(Path.home() / ".local/state")))
    return base / "Port-Operability-Metocean" / workspace_id / "progress.sqlite3"

def initialise(root: Path, grid: Path, countries: Path, first: int = 1979,
               last: int = 2025, selected_countries=None, database: Optional[Path] = None) -> None:
    periods = periods_for(first, last)
    frozen, land = gpd.read_file(grid), gpd.read_file(countries)
    sites = full.assign_continents(frozen, land)
    if selected_countries:
        missing = set(selected_countries) - set(sites.country)
        if missing:
            raise ValueError(f"No saved coastal points for countries: {sorted(missing)}")
        sites = sites.loc[sites.country.isin(selected_countries)].copy()
    sites = assign_regions(sites)
    if sites.empty:
        raise ValueError("Cannot initialise an empty coastal grid.")
    root = root.resolve()
    workspace_id = uuid.uuid4().hex
    database = (database or default_database(workspace_id)).resolve()
    if database.is_relative_to(root):
        raise ValueError("Keep the live SQLite database outside the workspace/cloud-sync directory.")
    if database.exists():
        raise FileExistsError(f"Refusing to overwrite an existing registry: {database}")
    root.mkdir(parents=True, exist_ok=False)
    inputs = root / "inputs"
    inputs.mkdir()
    shutil.copy2(grid, inputs / "source-coastal-grid.geojson")
    country_copy = inputs / countries.name
    shutil.copy2(countries, country_copy)
    atomic_csv(inputs / "sites.csv", sites)
    for name in ["downloads", "intermediate", "products", "progress", "logs", "quarantine"]:
        (root / name).mkdir()
    atomic_json(root / "workspace.json", {
        "schema_version": 1, "workspace_id": workspace_id, "created_at": now(),
        "database": str(database), "years": list(range(first, last + 1)),
        "periods": periods, "grid_sha256": model.file_hash(inputs / "source-coastal-grid.geojson"),
        "sites_sha256": model.file_hash(inputs / "sites.csv"),
        "countries_file": str(country_copy.relative_to(root)),
        "countries_sha256": model.file_hash(country_copy),
        "legacy_metocean_data_imported": False,
    })
    with Registry(root, create=True) as registry:
        registry.export()

class Registry:
    def __init__(self, root: Path, create: bool = False, read_only: bool = False):
        self.root = root.resolve()
        self.config = json.loads((self.root / "workspace.json").read_text())
        if self.config["schema_version"] != 1:
            raise ValueError("Unsupported workspace schema.")
        sites_file = self.root / "inputs/sites.csv"
        if model.file_hash(sites_file) != self.config["sites_sha256"]:
            raise ValueError("Frozen workspace grid changed; initialise a separate workspace.")
        self.sites = pd.read_csv(sites_file, float_precision="round_trip")
        self.database = Path(self.config["database"])
        if not create and not self.database.exists():
            raise FileNotFoundError("Local registry is missing. Run 'reconcile' to rebuild it from receipts.")
        self.database.parent.mkdir(parents=True, exist_ok=True)
        if create and read_only:
            raise ValueError("Cannot create a read-only registry.")
        self.db = (sqlite3.connect(self.database.as_uri() + "?mode=ro", uri=True, timeout=30)
                   if read_only else sqlite3.connect(self.database, timeout=30))
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        if not read_only:
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("PRAGMA synchronous=FULL")
        if create:
            self.db.executescript(SCHEMA)
            with self.db:
                self.db.execute("INSERT OR IGNORE INTO metadata VALUES ('workspace_id', ?)",
                                (self.config["workspace_id"],))
                self.db.executemany("INSERT OR IGNORE INTO sites VALUES (?,?,?,?,?,?)", [
                    (int(s.site_id), s.lon, s.lat, s.country, s.continent, s.region)
                    for s in self.sites.itertuples()])
                self.db.executemany(
                    "INSERT OR IGNORE INTO jobs(region,year,continent,updated_at) VALUES (?,?,?,?)",
                    [(region, year, group.continent.iloc[0], now())
                     for region, group in self.sites.groupby("region")
                     for year in self.config["years"]])
        identity = self.db.execute("SELECT value FROM metadata WHERE key='workspace_id'").fetchone()
        if identity is None or identity[0] != self.config["workspace_id"]:
            self.db.close()
            raise ValueError("Progress database belongs to another workspace.")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.db.close()

    @contextmanager
    def lock(self):
        import fcntl

        with self.database.with_suffix(".lock").open("a") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError("Another coordinator is already using this workspace.") from None
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def acquisition_root(self, region: str) -> Path:
        return self.root / "downloads" / region

    def group(self, region: str) -> pd.DataFrame:
        return self.sites.loc[self.sites.region.eq(region)].copy()

    def request_path(self, region: str, year: int) -> Path:
        continent = self.group(region).continent.iloc[0]
        return (self.acquisition_root(region) / "requests"
                / f"{continent.lower().replace(' ', '-')}-{year}.json")

    def method_metadata(self) -> dict:
        return {
            "version": 1, "science_sha256": model.file_hash(Path(model.__file__)),
            "processing_sha256": model.file_hash(Path(__file__).with_name("processing.py")),
            "scenarios": [asdict(s) for s in model.SCENARIOS],
            "max_ig_height": 14.6, "max_ig_period": 40,
        }

    def method_signature(self) -> str:
        return full._digest(self.method_metadata())

    def record_method(self) -> None:
        from importlib.metadata import version

        signature = self.method_signature()
        folder = self.root / "intermediate/methods" / signature
        if (folder / "method.json").exists():
            return
        folder.mkdir(parents=True, exist_ok=True)
        sources = [Path(model.__file__), Path(full.__file__), *Path(__file__).parent.glob("*.py")]
        hashes = {}
        for source in sources:
            relative = Path("operability") / source.name if source.parent == Path(__file__).parent else Path(source.name)
            target = folder / "source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            hashes[str(relative)] = model.file_hash(target)
        atomic_json(folder / "method.json", {
            "signature": signature, "settings": self.method_metadata(), "python": sys.version,
            "source_sha256": hashes, "packages": {name: version(name) for name in [
                "numpy", "pandas", "scipy", "xarray", "netCDF4", "dask", "distributed", "eccodes"]},
        })

    def progress(self) -> pd.DataFrame:
        frame = pd.read_sql_query("""
            SELECT s.*, COALESCE(p.months,0) extracted_months,
                   COALESCE(p.hours,0) cached_hours, COALESCE(p.valid,0) valid_hours,
                   COALESCE(p.retained,0) retained_months,
                   COALESCE(a.complete,0) completed_analyses,
                   COALESCE(a.blocked,0) blocked_analyses,
                   COALESCE(a.failed,0) failed_analyses,
                   COALESCE(a.needs_raw,0) needs_raw_analyses,
                   COALESCE(a.ready,0) ready_analyses
            FROM sites s LEFT JOIN (
                SELECT p.site_id,COUNT(*) months,SUM(expected_hours) hours,SUM(valid_hours) valid,
                       SUM(c.stamp != 'missing') retained
                FROM point_months p JOIN caches c USING(region,year,month)
                GROUP BY p.site_id) p USING(site_id)
            LEFT JOIN (
                SELECT site_id,SUM(status='complete') complete,SUM(status='blocked') blocked,
                       SUM(status='failed') failed,SUM(status='pending') ready,
                       SUM(status='waiting_for_raw') needs_raw
                FROM analyses WHERE method=? GROUP BY site_id) a USING(site_id)
            ORDER BY s.site_id
        """, self.db, params=(self.method_signature(),))
        frame["expected_months"] = 12 * len(self.config["years"])
        frame["expected_hours"] = sum((366 if calendar.isleap(y) else 365) * 24
                                      for y in self.config["years"])
        frame["expected_analyses"] = len(self.config["periods"])
        frame["extraction_status"] = np.select(
            [frame.extracted_months.eq(frame.expected_months), frame.extracted_months.gt(0)],
            ["complete", "partial"], default="missing")
        frame["source_gap_hours"] = frame.cached_hours - frame.valid_hours
        frame["extraction_fraction"] = frame.extracted_months / frame.expected_months
        frame["raw_status"] = np.select(
            [frame.retained_months.eq(frame.expected_months),
             frame.retained_months.eq(0) & frame.extracted_months.gt(0),
             frame.retained_months.gt(0)], ["complete", "removed", "partial"], default="missing")
        frame["analysis_status"] = np.select(
            [frame.completed_analyses.eq(frame.expected_analyses), frame.failed_analyses.gt(0),
             frame.needs_raw_analyses.gt(0), frame.blocked_analyses.gt(0),
             frame.completed_analyses.gt(0), frame.ready_analyses.gt(0)],
            ["complete", "failed", "needs_raw", "blocked", "partial", "ready"], default="waiting")
        return frame

    def export(self) -> None:
        atomic_csv(self.root / "progress/coastal-points.csv", self.progress())
        jobs = pd.read_sql_query("SELECT * FROM jobs ORDER BY year DESC,region", self.db)
        atomic_csv(self.root / "progress/download-jobs.csv", jobs)

    def failure_count(self, regions=None, years=None) -> int:
        jobs = self.db.execute("SELECT region,year FROM jobs WHERE status='failed'").fetchall()
        analyses = self.db.execute(
            "SELECT s.region,a.period FROM analyses a JOIN sites s USING(site_id) "
            "WHERE a.status='failed' AND a.method=?", (self.method_signature(),)).fetchall()
        return (sum((regions is None or r["region"] in regions)
                    and (years is None or r["year"] in years) for r in jobs)
                + sum((regions is None or r["region"] in regions)
                      and (years is None or set(period_years(self.config["periods"][r["period"]])) <= set(years))
                      for r in analyses))

    def quarantine(self, paths) -> None:
        for path in paths:
            if path.exists():
                target = self.root / "quarantine" / (
                    path.name + "." + uuid.uuid4().hex[:12])
                LOGGER.warning("Preserving unverified/corrupt file in %s", target)
                path.replace(target)

    def coverage_path(self, region: str, year: int) -> Path:
        return self.root / "intermediate/coverage" / region / f"{year}.json.gz"

    def restore_coverage(self, region: str, year: int) -> None:
        path = self.coverage_path(region, year)
        if not path.exists():
            return
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            history = json.load(handle)
        if (history["workspace_id"] != self.config["workspace_id"]
                or history["grid_digest"] != full.grid_digest(self.group(region))):
            raise ValueError(f"Coverage history belongs to another workspace/grid: {path}")
        count = self.db.execute("SELECT COUNT(*) FROM point_months WHERE region=? AND year=?",
                                (region, year)).fetchone()[0]
        if count == sum(len(record["points"]) for record in history["months"].values()):
            return
        with self.db:
            for month, record in history["months"].items():
                month = int(month)
                self.db.execute("INSERT OR IGNORE INTO caches VALUES (?,?,?,?,?,?)",
                                (region, year, month, record["path"], record["sha256"], "missing"))
                self.db.executemany("INSERT OR IGNORE INTO point_months VALUES (?,?,?,?,?,?,?)", [
                    (p["site_id"], region, year, month, p["expected_hours"],
                     p["valid_hours"], p["variable_hours"]) for p in record["points"]])

    def persist_coverage(self, region: str, year: int) -> None:
        months = {}
        for cache in self.db.execute(
                "SELECT * FROM caches WHERE region=? AND year=? ORDER BY month", (region, year)):
            points = [dict(p) for p in self.db.execute(
                "SELECT site_id,expected_hours,valid_hours,variable_hours FROM point_months "
                "WHERE region=? AND year=? AND month=? ORDER BY site_id", (region, year, cache["month"]))]
            months[str(cache["month"])] = {
                "path": cache["path"], "sha256": cache["sha256"], "points": points}
        if not months:
            return
        path, manifest = self.coverage_path(region, year), self.request_path(region, year)
        old = {}
        if path.exists():
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                old = json.load(handle)
        history = {
            "workspace_id": self.config["workspace_id"], "region": region, "year": year,
            "grid_digest": full.grid_digest(self.group(region)), "months": months,
            "request_manifest": json.loads(manifest.read_text()) if manifest.exists()
            else old.get("request_manifest"),
        }
        if history != old:
            atomic_gzip_json(path, history)

    def reconcile(self, regions=None, repair: bool = False, years=None) -> int:
        """Only new/changed files are rehashed; receipts can rebuild a lost database."""
        errors = 0
        for region, group in self.sites.groupby("region"):
            if regions is not None and region not in regions:
                continue
            continent = group.continent.iloc[0]
            for year in self.config["years"]:
                if years is not None and year not in years:
                    continue
                self.restore_coverage(region, year)
                job = self.db.execute("SELECT * FROM jobs WHERE region=? AND year=?",
                                      (region, year)).fetchone()
                problem = None
                for month in range(1, 13):
                    path = full.month_path(self.acquisition_root(region), continent, year, month)
                    receipt = path.with_suffix(".nc.source.json")
                    previous = self.db.execute(
                        "SELECT * FROM caches WHERE region=? AND year=? AND month=?",
                        (region, year, month)).fetchone()
                    if not path.exists() or not receipt.exists():
                        if path.exists():
                            LOGGER.warning("Interrupted cache has no receipt: %s", path)
                            self.quarantine([path])
                        with self.db:
                            self.db.execute("UPDATE caches SET stamp='missing' "
                                            "WHERE region=? AND year=? AND month=?",
                                            (region, year, month))
                        continue
                    stamp = file_stamp([path, receipt])
                    if previous is not None and previous["stamp"] == stamp:
                        continue
                    try:
                        source = json.loads(receipt.read_text())
                        if (source["grid_digest"] != full.grid_digest(group)
                                or source["sha256"] != model.file_hash(path)):
                            raise ValueError("Cache checksum or grid identity differs from its receipt.")
                        with xr.open_dataset(path) as ds:
                            full._check_month(ds, group, year, month)
                            values = {v: ds[v].values for v in full.VARIABLES}
                            valid = np.logical_and.reduce([np.isfinite(v) for v in values.values()])
                            valid &= (values["swh"] >= 0) & (values["mwp"] > 0) & (values["pp1d"] > 0)
                            counts = {v: np.isfinite(a).sum(axis=0) for v, a in values.items()}
                            hours, valid_hours = ds.sizes["time"], valid.sum(axis=0)
                        with self.db:
                            self.db.execute("DELETE FROM caches WHERE region=? AND year=? AND month=?",
                                            (region, year, month))
                            self.db.execute("INSERT INTO caches VALUES (?,?,?,?,?,?)",
                                            (region, year, month, str(path.relative_to(self.root)),
                                             source["sha256"], stamp))
                            self.db.executemany("INSERT INTO point_months VALUES (?,?,?,?,?,?,?)", [
                                (int(site), region, year, month, hours, int(valid_hours[i]),
                                 json.dumps({v: int(a[i]) for v, a in counts.items()}))
                                for i, site in enumerate(group.site_id)])
                    except (ValueError, KeyError, OSError) as error:
                        LOGGER.error("Cache validation failed for %s: %s", path, error)
                        with self.db:
                            self.db.execute("UPDATE caches SET stamp='missing' "
                                            "WHERE region=? AND year=? AND month=?",
                                            (region, year, month))
                        if repair:
                            self.quarantine([path, receipt])
                        else:
                            problem = f"Cache validation failed for {path}: {error}"
                            errors += 1
                manifest = self.request_path(region, year)
                request = json.loads(manifest.read_text()) if manifest.exists() else {}
                if repair and request.get("status") in {"failed", "dismissed", "deleted"}:
                    archive = manifest.parent / "retired" / f"{year}-{request['request_id']}.json"
                    archive.parent.mkdir(parents=True, exist_ok=True)
                    manifest.replace(archive)
                    LOGGER.warning("Archived failed CDS job; a new request will consume budget: %s", archive)
                    request = {}
                cached = self.db.execute("SELECT COUNT(*) FROM caches WHERE region=? AND year=?",
                                         (region, year)).fetchone()[0]
                if problem:
                    status = "failed"
                elif cached == 12:
                    status = "complete"
                elif request.get("status") in {"accepted", "running"}:
                    status = "submitted"
                elif request.get("status") in {"failed", "dismissed", "deleted"}:
                    status, problem = "failed", f"CDS job ended with status {request['status']}."
                elif job["status"] == "failed" and not repair:
                    status, problem = "failed", job["error"]
                else:
                    status = "partial" if cached else "pending"
                with self.db:
                    self.db.execute("""UPDATE jobs SET status=?,request_id=?,error=?,updated_at=?
                                       WHERE region=? AND year=?""",
                                    (status, request.get("request_id"), problem, now(), region, year))
                self.persist_coverage(region, year)
            self.refresh_analyses(region, retry=repair)
            waiting = [r[0] for r in self.db.execute(
                "SELECT DISTINCT a.period FROM analyses a JOIN sites s USING(site_id) "
                "WHERE s.region=? AND a.status='waiting_for_raw'", (region,))]
            needed_years = {y for period in waiting for y in period_years(self.config["periods"][period])}
            with self.db:
                for year in self.config["years"]:
                    available = self.db.execute(
                        "SELECT COUNT(*) FROM caches WHERE region=? AND year=? AND stamp != 'missing'",
                        (region, year)).fetchone()[0]
                    if available < 12:
                        self.db.execute(
                            "UPDATE jobs SET status=? WHERE region=? AND year=? AND status='complete'",
                            ("partial" if year in needed_years else "archived", region, year))
        self.export()
        return max(errors, self.failure_count(regions, years))

    def refresh_analyses(self, region: str, retry: bool = False) -> None:
        group, method = self.group(region), self.method_signature()
        ready = {}
        for period, bounds in self.config["periods"].items():
            years = period_years(bounds)
            caches = self.db.execute(
                "SELECT year,month,sha256,stamp FROM caches WHERE region=? AND year BETWEEN ? AND ? "
                "ORDER BY year,month", (region, years[0], years[-1])).fetchall()
            if len(caches) != len(years) * 12:
                with self.db:
                    self.db.execute("""UPDATE analyses SET status='waiting',updated_at=?
                        WHERE period=? AND site_id IN (SELECT site_id FROM sites WHERE region=?)""",
                                    (now(), period, region))
                continue
            inputs = full._digest([(r["year"], r["month"], r["sha256"]) for r in caches])
            retained = all(r["stamp"] != "missing" for r in caches)
            counts = dict(self.db.execute(
                "SELECT site_id,SUM(valid_hours) FROM point_months WHERE region=? "
                "AND year BETWEEN ? AND ? GROUP BY site_id", (region, years[0], years[-1])))
            ready[period] = (inputs, counts, retained)
        for site in group.itertuples(index=False):
            path = product_path(self.root, site.site_id)
            archive, archive_error = None, None
            if path.exists():
                try:
                    archive = read_archive(path)
                except (ValueError, KeyError, OSError, EOFError) as failure:
                    LOGGER.error("Invalid point archive %s: %s", path, failure)
                    if retry:
                        self.quarantine([path])
                    else:
                        archive_error = str(failure)
            if archive is not None:
                export_tables(self.root, site.site_id, archive)
            for period, (inputs, counts, retained) in ready.items():
                old = self.db.execute("SELECT * FROM analyses WHERE site_id=? AND period=?",
                                      (int(site.site_id), period)).fetchone()
                expected = expected_product(self.config, site.site_id, period, method, inputs)
                error, stamp = None, None
                if int(site.site_id) not in counts:
                    status, error = "waiting_for_raw", "Coverage metadata is missing."
                elif counts[int(site.site_id)] == 0:
                    status, error = "blocked", "No jointly valid source observations; not a download failure."
                else:
                    try:
                        if archive_error:
                            raise ValueError(archive_error)
                        stamp = verified_product(self.root, expected, old, archive=archive)
                        if stamp:
                            status = "complete"
                        elif (old is not None and old["status"] == "failed" and not retry
                              and old["method"] == method and old["inputs"] == inputs):
                            status, error = "failed", old["error"]
                        else:
                            status = "pending" if retained else "waiting_for_raw"
                            if not retained:
                                error = "Raw files removed; retained products are independent, but recalculation needs raw input."
                    except (ValueError, KeyError, OSError) as failure:
                        LOGGER.error("Invalid product for site %s/%s: %s", site.site_id, period, failure)
                        if retry:
                            self.quarantine([path])
                            archive = None
                        status, error = ("pending" if retry else "failed"), str(failure)
                with self.db:
                    self.db.execute("INSERT OR REPLACE INTO analyses VALUES (?,?,?,?,?,?,?,?)",
                                    (int(site.site_id), period, status, method, inputs,
                                     stamp, error, now()))


