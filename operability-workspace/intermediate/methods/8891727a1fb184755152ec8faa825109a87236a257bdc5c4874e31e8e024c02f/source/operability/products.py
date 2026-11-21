"""Durable analysis checkpoints and portable CSV products, independent of raw data."""

from __future__ import annotations

from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import logging
import os
from pathlib import Path

import pandas as pd

import port_operability as model

LOGGER = logging.getLogger("operability")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.part")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.part")
    compression = {"method": "gzip", "mtime": 0} if path.suffix == ".gz" else None
    frame.to_csv(temporary, index=False, compression=compression)
    temporary.replace(path)


def atomic_gzip_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.part")
    with temporary.open("wb") as handle, gzip.GzipFile(
            filename="", mode="wb", fileobj=handle, mtime=0) as compressed:
        compressed.write(json.dumps(value, allow_nan=False).encode())
    temporary.replace(path)


def file_stamp(paths) -> str:
    return json.dumps([(p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ctime_ns) for p in paths])


def product_path(root: Path, site_id: int) -> Path:
    return root / "products" / f"site-{int(site_id)}.json.gz"


def read_archive(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        archive = json.load(handle)
    if not isinstance(archive, dict) or not isinstance(archive.get("periods"), dict):
        raise ValueError(f"Invalid point archive structure: {path}")
    for receipt in archive["periods"].values():
        if (not isinstance(receipt, dict) or not isinstance(receipt.get("tables"), dict)
                or not isinstance(receipt.get("sha256"), dict) or "expected" not in receipt):
            raise ValueError(f"Invalid point receipt structure: {path}")
        if set(receipt["tables"]) != {"summary", "annual", "seasonal"}:
            raise ValueError(f"Calculated archive is missing tables: {path}")
        if any(not isinstance(text, str)
               or hashlib.sha256(text.encode()).hexdigest() != receipt["sha256"].get(name)
               for name, text in receipt["tables"].items()):
            raise ValueError(f"Calculated product checksum mismatch: {path}")
    return archive


def save_product(root: Path, locks: Path, expected: dict, tables: dict) -> None:
    import fcntl

    site_id = expected["site_id"]
    path = product_path(root, site_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    locks.mkdir(parents=True, exist_ok=True)
    # Different periods for the same point may finish in different workers.
    with (locks / f"{site_id}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        archive = read_archive(path) if path.exists() else {"site_id": site_id, "periods": {}}
        archive["periods"][expected["period"]] = {
            "expected": expected, "tables": tables,
            "sha256": {name: hashlib.sha256(text.encode()).hexdigest() for name, text in tables.items()},
            "completed_at": now(), "writer_sha256": model.file_hash(Path(__file__)),
            "exact_original_reproduction": False,
        }
        atomic_gzip_json(path, archive)
        export_tables(root, site_id, archive)


def export_tables(root: Path, site_id: int, archive: dict) -> None:
    """Keep standard tables independent of the raw input and the local database."""
    if not archive["periods"]:
        return
    folder = root / "products/tables" / f"site-{int(site_id)}"
    manifest = folder / "provenance.json"
    paths = [folder / f"{name}.csv.gz" for name in ["summary", "annual", "seasonal"]]
    archive_sha = model.file_hash(product_path(root, site_id))
    if manifest.exists() and all(path.exists() for path in paths):
        previous = json.loads(manifest.read_text())
        if previous["archive_sha256"] == archive_sha and previous["stamp"] == file_stamp(paths):
            return
    LOGGER.info("Writing portable analysis tables for site %s", site_id)
    hashes = {}
    for path in paths:
        name = path.name.split(".")[0]
        frames = [
            pd.read_csv(io.StringIO(receipt["tables"][name])).assign(
                method_signature=receipt["expected"]["method"],
                source_inputs=receipt["expected"]["inputs"])
            for receipt in archive["periods"].values()]
        atomic_csv(path, pd.concat(frames, ignore_index=True))
        hashes[path.name] = model.file_hash(path)
    atomic_json(manifest, {
        "site_id": int(site_id), "archive_sha256": archive_sha, "sha256": hashes,
        "periods": {name: receipt["expected"] for name, receipt in archive["periods"].items()},
        "method_metadata_directory": "intermediate/methods/<method_signature>/",
        "units": {"availability": "percent", "duration": "hours",
                  "frequency": "events per 365.25 observed days",
                  "mean_period": "seconds", "peak_period": "seconds",
                  "coordinates": "degrees, EPSG:4326", "depth": "metres"},
        "stamp": file_stamp(paths), "raw_inputs_required_to_read": False,
        "exact_original_reproduction": False,
    })


def expected_product(config: dict, site_id: int, period: str, method: str, inputs: str) -> dict:
    return {"workspace_id": config["workspace_id"], "site_id": int(site_id),
            "period": period, "bounds": config["periods"][period],
            "method": method, "inputs": inputs}


def verified_product(root: Path, expected: dict, previous=None, archive=None):
    path = product_path(root, expected["site_id"])
    if not path.exists():
        return None
    stamp = file_stamp([path])
    if (previous is not None and previous["status"] == "complete"
            and previous["method"] == expected["method"] and previous["inputs"] == expected["inputs"]
            and previous["stamp"] == stamp):
        return stamp
    contents = archive if archive is not None else read_archive(path)
    receipt = contents["periods"].get(expected["period"])
    if receipt is None:
        return None
    if receipt["expected"] != expected:
        return None
    if set(receipt["tables"]) != {"summary", "annual", "seasonal"}:
        raise ValueError(f"Calculated archive is missing tables: {path}")
    if any(hashlib.sha256(text.encode()).hexdigest() != receipt["sha256"][name]
           for name, text in receipt["tables"].items()):
        raise ValueError(f"Calculated product checksum mismatch: {path}")
    return stamp
