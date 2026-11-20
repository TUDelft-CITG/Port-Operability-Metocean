"""Sequential CDS acquisition and bounded process-based Dask calculations."""

from __future__ import annotations

import calendar
from contextlib import contextmanager
import json
import logging
import math
import os
from pathlib import Path
import sys
import threading
import time

import pandas as pd
import xarray as xr

import full_study as full
import port_operability as model
from .workspace import Registry, period_years
from .products import atomic_json, expected_product, now, save_product, verified_product

LOGGER = logging.getLogger("operability")


@contextmanager
def watch_request(registry: Registry, region: str, year: int):
    """Keep administration live while the sequential CDS helper waits remotely."""
    stopped, errors = threading.Event(), []
    manifest = registry.request_path(region, year)

    def monitor():
        try:
            with Registry(registry.root) as observer:
                previous = None
                while not stopped.is_set():
                    if manifest.exists():
                        try:
                            request = json.loads(manifest.read_text())
                        except json.JSONDecodeError:
                            LOGGER.warning("Request manifest is being written; will retry: %s", manifest)
                        else:
                            identity = (request.get("request_id"), request.get("status"))
                            if identity != previous:
                                status = ("extracting" if request.get("status") == "successful"
                                          else "failed" if request.get("status") in {"failed", "dismissed", "deleted"}
                                          else "submitted")
                                with observer.db:
                                    observer.db.execute(
                                        "UPDATE jobs SET status=?,request_id=?,updated_at=? WHERE region=? AND year=?",
                                        (status, request.get("request_id"), now(), region, year))
                                observer.export()
                                LOGGER.info("Acquisition %s/%s: %s (CDS job %s)",
                                            region, year, status, request.get("request_id"))
                                previous = identity
                    stopped.wait(.5)
        except Exception as error:
            LOGGER.exception("Live acquisition administration failed for %s/%s", region, year)
            errors.append(error)

    thread = threading.Thread(target=monitor, name="acquisition-progress")
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join()
        if errors and sys.exc_info()[0] is None:
            raise RuntimeError("Acquisition progress monitor failed; request receipts are retained.") from errors[0]


def prepare_lookups(root: Path) -> None:
    for depth in sorted({s.depth_m for s in model.SCENARIOS}):
        model.ig_lookup(root / "intermediate" / f"ig-{depth:g}m.nc", depth,
                        max_height=14.6, max_period=40)

def calculate_batch(spec: dict) -> list[dict]:
    """Workers own no database connections and write only their distinct point products."""
    root = Path(spec["root"])
    group = pd.DataFrame(spec["sites"])
    bounds = spec["bounds"]
    years = period_years(bounds)
    ds = full._load_batch(group, spec["continent"], years, range(1, 13),
                          Path(spec["acquisition_root"]))
    lookups = {}
    for depth in sorted({s.depth_m for s in model.SCENARIOS}):
        with xr.open_dataset(root / "intermediate" / f"ig-{depth:g}m.nc") as table:
            lookups[depth] = table.load()
    results = []
    for _, site in group.iterrows():
        expected = spec["expected"][str(int(site.site_id))]
        try:
            if verified_product(root, expected):
                results.append({"site_id": int(site.site_id), "status": "complete"})
                continue
            frames = {"summary": [], "annual": [], "seasonal": []}
            point = ds.sel(site=int(site.site_id))
            for scenario in model.SCENARIOS:
                frame = model.prepare_site(point, site, lookups[scenario.depth_m], scenario)
                frame = frame.loc[(frame.index >= bounds[0]) & (frame.index < bounds[1])]
                summary, annual, seasonal = model.summarise_frame(frame, site, scenario)
                expected_hours = int((pd.Timestamp(bounds[1]) - pd.Timestamp(bounds[0])).total_seconds() / 3600)
                summary["expected_hours"] = expected_hours
                summary["coverage_fraction"] = summary.observed_hours / expected_hours
                annual["expected_hours"] = annual.year.map(
                    lambda y: (366 if calendar.isleap(y) else 365) * 24)
                annual["complete_year"] = annual.observed_hours.eq(annual.expected_hours)
                for name, result in zip(frames, (summary, annual, seasonal)):
                    frames[name].append(result.assign(
                        period=expected["period"], country=site.country, continent=site.continent))
            tables = {name: pd.concat(parts, ignore_index=True).to_csv(index=False)
                      for name, parts in frames.items()}
            save_product(root, Path(spec["locks"]), expected, tables)
            results.append({"site_id": int(site.site_id), "status": "complete"})
        except Exception as error:
            LOGGER.exception("Calculation failed for site %s/%s", site.site_id, expected["period"])
            results.append({"site_id": int(site.site_id), "status": "failed",
                            "error": f"{type(error).__name__}: {error}"})
    return results

def ready_batches(registry: Registry, continents, years, workers: int, batch_size: int) -> list[dict]:
    rows = pd.read_sql_query("""SELECT a.*,s.region,s.continent FROM analyses a JOIN sites s
        USING(site_id) WHERE a.status='pending' ORDER BY a.period DESC,s.region,a.site_id""", registry.db)
    if continents:
        rows = rows.loc[rows.continent.isin(continents)]
    specs = []
    for (period, region), group in rows.groupby(["period", "region"], sort=False):
        bounds = registry.config["periods"][period]
        if years and not set(period_years(bounds)) <= set(years):
            continue
        for start in range(0, len(group), batch_size):
            part = group.iloc[start:start + batch_size]
            selected = registry.sites.loc[registry.sites.site_id.isin(part.site_id)]
            specs.append({
                "root": str(registry.root), "sites": selected.to_dict("records"),
                "locks": str(registry.database.parent / "point-locks"),
                "continent": selected.continent.iloc[0], "bounds": bounds, "period": period,
                "acquisition_root": str(registry.acquisition_root(region)),
                "expected": {str(r.site_id): expected_product(
                    registry.config, r.site_id, period, r.method, r.inputs) for r in part.itertuples()},
            })
            if len(specs) == workers:
                return specs
    return specs

def next_job(registry: Registry, continents, years, allow_new: bool):
    for job in registry.db.execute(
            "SELECT * FROM jobs WHERE status IN ('pending','partial','submitted') ORDER BY year DESC,region"):
        if continents and job["continent"] not in continents:
            continue
        if years and job["year"] not in years:
            continue
        if allow_new or registry.request_path(job["region"], job["year"]).exists():
            return job
    return None

def run(registry: Registry, hours: float = 2, workers: int = 2, batch_size: int = 8,
        max_requests: int = 4, stage: str = "all", continents=None, years=None,
        retry_failed: bool = False, dashboard_address: str = "127.0.0.1:8787",
        memory_limit: str = "2GiB") -> int:
    from distributed import Client, LocalCluster, as_completed

    if not math.isfinite(hours) or hours < 0 or workers < 1 or batch_size < 1 or max_requests < 0:
        raise ValueError("Nonnegative finite hours/request budget and positive workers/batch size required.")
    if stage not in {"all", "download", "calculate"}:
        raise ValueError("Stage must be all, download or calculate.")
    if continents and not set(continents) <= set(registry.sites.continent):
        raise ValueError("Unknown or empty continent selection.")
    if years and not set(years) <= set(registry.config["years"]):
        raise ValueError("Years must be inside the initialised workspace plan.")
    deadline, requests_used, failures = time.monotonic() + hours * 3600, 0, 0
    session_file = registry.root / "progress/session.json"
    with registry.lock():
        selected_regions = set(registry.sites.loc[
            registry.sites.continent.isin(continents) if continents
            else registry.sites.site_id.notna(), "region"])
        registry.reconcile(regions=selected_regions, repair=retry_failed, years=years)
        failures = registry.failure_count(selected_regions, years)
        if failures:
            LOGGER.error("%d selected tasks remain failed; inspect progress or use --retry-failed.", failures)
        if time.monotonic() >= deadline:
            LOGGER.info("Time budget exhausted; no work submitted.")
            return int(failures > 0)
        session = {"started_at": now(), "pid": os.getpid(), "status": "running",
                   "hours": hours, "workers": workers}
        try:
            with LocalCluster(n_workers=workers, threads_per_worker=1, processes=True,
                              host="127.0.0.1", dashboard_address=dashboard_address,
                              memory_limit=memory_limit,
                              local_directory=str(registry.database.parent / "dask")) as cluster, \
                    Client(cluster) as client:
                client.wait_for_workers(workers)
                session["dashboard"] = client.dashboard_link
                atomic_json(session_file, session)
                LOGGER.info("Dask dashboard: %s (%d single-threaded process workers)",
                            client.dashboard_link, len(client.scheduler_info()["workers"]))
                lookups_ready = False
                while time.monotonic() < deadline:
                    specs = ([] if stage == "download" else ready_batches(
                        registry, continents, years, workers, batch_size))
                    if specs:
                        if not lookups_ready:
                            registry.record_method()
                            prepare_lookups(registry.root)
                            lookups_ready = True
                            if time.monotonic() >= deadline:
                                break
                        futures = {}
                        for spec in specs:
                            with registry.db:
                                registry.db.executemany(
                                    "UPDATE analyses SET status='running',updated_at=? "
                                    "WHERE site_id=? AND period=?",
                                    [(now(), int(s["site_id"]), spec["period"]) for s in spec["sites"]])
                            futures[client.submit(calculate_batch, spec, pure=False)] = spec
                        for future in as_completed(futures):
                            spec = futures[future]
                            try:
                                results = future.result()
                            except Exception as error:
                                LOGGER.exception("Dask batch failed for %s", spec["period"])
                                results = [{"site_id": s["site_id"], "status": "failed",
                                            "error": f"{type(error).__name__}: {error}"}
                                           for s in spec["sites"]]
                            for result in results:
                                site_id = int(result["site_id"])
                                stamp = (verified_product(registry.root, spec["expected"][str(site_id)])
                                         if result["status"] == "complete" else None)
                                if result["status"] == "complete" and stamp is None:
                                    raise RuntimeError("Worker reported completion without verified products.")
                                with registry.db:
                                    registry.db.execute(
                                        "UPDATE analyses SET status=?,stamp=?,error=?,updated_at=? "
                                        "WHERE site_id=? AND period=?",
                                        (result["status"], stamp, result.get("error"), now(),
                                         site_id, spec["period"]))
                                if result["status"] == "failed":
                                    failures += 1
                                    LOGGER.error("Site %s/%s failed: %s",
                                                 site_id, spec["period"], result["error"])
                            registry.export()
                            future.release()
                        continue
                    if stage == "calculate":
                        break
                    job = next_job(registry, continents, years, requests_used < max_requests)
                    if job is None:
                        break
                    region, year = job["region"], job["year"]
                    manifest = registry.request_path(region, year)
                    new_request = not manifest.exists()
                    try:
                        with watch_request(registry, region, year):
                            full.download_grib_years(
                                registry.group(region), [job["continent"]], [year],
                                registry.acquisition_root(region),
                                request_budget=int(new_request),
                                max_wait_seconds=max(1, math.ceil(deadline - time.monotonic())))
                    except TimeoutError:
                        LOGGER.info("CDS wait reached the session deadline; saved job will resume.")
                        registry.reconcile(regions=[region], years=[year])
                        break
                    except Exception as error:
                        LOGGER.exception("Acquisition failed for %s/%s", region, year)
                        registry.reconcile(regions=[region], years=[year])
                        with registry.db:
                            registry.db.execute(
                                "UPDATE jobs SET status='failed',error=?,updated_at=? WHERE region=? AND year=?",
                                (f"{type(error).__name__}: {error}", now(), region, year))
                        registry.export()
                        failures += 1
                        break
                    finally:
                        requests_used += int(new_request and manifest.exists())
                    failures += registry.reconcile(regions=[region], years=[year])
                session["status"] = "finished" if not failures else "finished_with_errors"
        except KeyboardInterrupt:
            session["status"] = "interrupted"
            LOGGER.warning("Interrupted; verified files and submitted job IDs are retained.")
            raise
        except Exception:
            session["status"] = "failed"
            raise
        finally:
            session["finished_at"] = now()
            atomic_json(session_file, session)
            registry.export()
    LOGGER.info("Session stopped: %d new requests; %d failures. Progress: %s",
                requests_used, failures, registry.root / "progress/coastal-points.csv")
    return int(failures > 0)
