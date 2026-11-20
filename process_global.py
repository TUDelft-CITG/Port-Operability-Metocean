"""Command-line entry point for the resumable global operability workspace."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import time

import full_study as full
from operability.workspace import DEFAULT_WORKSPACE, Registry, initialise
from operability.processing import run

LOGGER = logging.getLogger("operability")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="Freeze static inputs; no metocean download.")
    init.add_argument("--grid", type=Path, default=Path("data/processed/coastal-grid.geojson"))
    init.add_argument("--countries-file", type=Path,
                      default=Path("data/reference/ne_110m_admin_0_countries.zip"))
    init.add_argument("--start-year", type=int, default=1979)
    init.add_argument("--end-year", type=int, default=2025)
    init.add_argument("--countries", nargs="+")
    init.add_argument("--database", type=Path)
    commands.add_parser("status", help="Display the durable central administration.")
    reconcile = commands.add_parser("reconcile", help="Recover progress from verified files and receipts.")
    reconcile.add_argument("--retry-failed", action="store_true",
                           help="Quarantine corrupt files; allow deliberate retry of failed jobs/products.")
    execute = commands.add_parser("run", help="Run a bounded session with a live local Dask dashboard.")
    execute.add_argument("--hours", type=float, default=2)
    execute.add_argument("--workers", type=int, default=2)
    execute.add_argument("--batch-size", type=int, default=8)
    execute.add_argument("--max-requests", type=int, default=4)
    execute.add_argument("--stage", choices=["all", "download", "calculate"], default="all")
    execute.add_argument("--continents", nargs="+", choices=full.CONTINENTS)
    execute.add_argument("--years", nargs="+", type=int)
    execute.add_argument("--retry-failed", action="store_true")
    execute.add_argument("--dashboard-port", type=int, default=8787)
    execute.add_argument("--worker-memory", default="2GiB")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.command == "init":
        initialise(args.workspace, args.grid, args.countries_file, args.start_year, args.end_year,
                   args.countries, args.database)
    else:
        config = json.loads((args.workspace / "workspace.json").read_text())
        rebuilding = args.command == "reconcile" and not Path(config["database"]).exists()
        if rebuilding:
            LOGGER.warning("Rebuilding missing local database from workspace receipts.")
        with Registry(args.workspace, create=rebuilding) as registry:
            if args.command != "status":
                log_directory = registry.database.parent / "logs"
                log_directory.mkdir(parents=True, exist_ok=True)
                handler = logging.FileHandler(
                    log_directory / f"session-{time.time_ns()}.log")
                handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
                LOGGER.addHandler(handler)
                try:
                    if args.command == "reconcile":
                        with registry.lock():
                            return int(registry.reconcile(repair=args.retry_failed) > 0)
                    return run(
                        registry, args.hours, args.workers, args.batch_size, args.max_requests,
                        args.stage, args.continents, args.years, args.retry_failed,
                        f"127.0.0.1:{args.dashboard_port}", args.worker_memory)
                finally:
                    LOGGER.removeHandler(handler)
                    handler.close()
            print(registry.progress().groupby(
                ["continent", "extraction_status", "analysis_status"]).size().to_string())
    with Registry(args.workspace) as registry:
        print(f"\nWorkspace: {registry.root}\nLocal database: {registry.database}")
        print(f"Coastal points: {len(registry.sites):,}; years: "
              f"{registry.config['years'][0]}-{registry.config['years'][-1]}")
        print("No legacy ERA5 payloads were imported. See progress/coastal-points.csv.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
