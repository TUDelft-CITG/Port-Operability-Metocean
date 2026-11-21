"""Read-only loading and plotting of existing workspace results."""

from __future__ import annotations

import io
import json
from html import escape
from pathlib import Path
from typing import Optional

import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

import port_operability as model
from .workspace import Registry
from .products import expected_product, read_archive, verified_product

RISK_COLOURS = {"Low": "#00cc00", "Medium": "#ffff33",
                "High": "#ff7f50", "Very high": "#d73027"}
PROGRESS_COLOURS = {"missing": "#969696", "partial": "#e6b800",
                    "complete": "#238b8e", "complete with source gaps": "#d95f02"}


def read_progress(root: Path) -> pd.DataFrame:
    with Registry(root, read_only=True) as registry:
        return registry.progress()


def available_periods(root: Path) -> pd.DataFrame:
    records = []
    for path in sorted((root / "products/tables").glob("site-*/provenance.json")):
        provenance = json.loads(path.read_text())
        records.extend({"period": name, "method_signature": p["method"]}
                       for name, p in provenance["periods"].items())
    if not records:
        return pd.DataFrame(columns=["period", "method_signature", "completed_points"])
    return pd.DataFrame(records).groupby(["period", "method_signature"]).size().rename(
        "completed_points").reset_index().sort_values("period", ascending=False)


def load_land(root: Path) -> gpd.GeoDataFrame:
    config = json.loads((root / "workspace.json").read_text())
    path = root / config["countries_file"]
    if model.file_hash(path) != config["countries_sha256"]:
        raise ValueError("Frozen country reference changed.")
    return gpd.read_file(path)


def plot_progress(progress: pd.DataFrame):
    counts = progress.groupby(["continent", "extraction_status"]).size().unstack(fill_value=0)
    figure, axis = plt.subplots(figsize=(11, 4))
    counts.reindex(columns=["missing", "partial", "complete"], fill_value=0).plot.bar(
        stacked=True, ax=axis, color=[PROGRESS_COLOURS[k] for k in ["missing", "partial", "complete"]])
    axis.set(xlabel="", ylabel="Coastal points", title="Verified extraction coverage")
    axis.tick_params(axis="x", rotation=20)
    figure.tight_layout()
    return figure


def plot_availability(summary: pd.DataFrame, seasonal: pd.DataFrame, scenario: str = "baseline"):
    selected = summary.loc[summary.scenario.eq(scenario)]
    if selected.empty:
        raise ValueError(f"No calculated results for scenario {scenario}.")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(selected.total_availability, bins=range(0, 105, 5), color="#238b8e")
    axes[0].set(xlabel="Conditional availability (%)", ylabel="Positions", xlim=(0, 100))
    seasonal.loc[seasonal.scenario.eq(scenario)].groupby("season").total_availability.mean().reindex(
        ["DJF", "MAM", "JJA", "SON"]).plot.bar(ax=axes[1], color="#238b8e", rot=0)
    axes[1].set(ylabel="Mean conditional availability (%)", xlabel="", ylim=(0, 100))
    figure.tight_layout()
    return figure


def make_map(points: pd.DataFrame, countries: gpd.GeoDataFrame, kind: str = "progress",
             scenario: str = "baseline") -> folium.Map:
    """Return an embedded, tile-free progress or calculated-result map; never export HTML."""
    if kind not in {"progress", "availability"}:
        raise ValueError("Map kind must be progress or availability.")
    points = points.copy()
    if kind == "availability":
        points = points.loc[points.scenario.eq(scenario)]
        colours, title = RISK_COLOURS, "Calculated operability risk"
        points["map_class"] = points.risk
    else:
        colours, title = PROGRESS_COLOURS, "Extraction progress"
        points["map_class"] = points.extraction_status
        points.loc[points.extraction_status.eq("complete") & points.source_gap_hours.gt(0),
                   "map_class"] = "complete with source gaps"
    if points.empty or points.site_id.duplicated().any():
        raise ValueError("Select one nonempty period/scenario with unique coastal points.")
    result = folium.Map(location=[15, 0], zoom_start=2, tiles=None, prefer_canvas=True)
    result.get_root().header.add_child(folium.Element(
        "<style>.leaflet-container {background:#bfd2d9 !important;}</style>"))
    folium.GeoJson(
        json.loads(countries[["ADMIN", "geometry"]].to_json()),
        name="Natural Earth land", overlay=False, interactive=False,
        attribution="Natural Earth, public domain",
        style_function=lambda feature: {"fillColor": "#eeefee", "fillOpacity": 1,
                                         "color": "#888888", "weight": .5},
    ).add_to(result)
    for label, colour in colours.items():
        layer = folium.FeatureGroup(name=label)
        for point in points.loc[points.map_class.eq(label)].itertuples(index=False):
            popup = f"<b>Site {point.site_id}</b><br>{escape(point.country)}<br>"
            if kind == "availability":
                popup += (
                    f"Period: {escape(point.period)}<br>Scenario: {escape(scenario)}<br>"
                    f"Availability: {point.total_availability:.2f}%<br>Risk: {escape(label)}<br>"
                    f"Main driver: {escape(point.main_driver)}<br>"
                    f"Wind: {point.wind_availability:.2f}%<br>"
                    f"Short waves: {point.short_wave_availability:.2f}%<br>"
                    f"IG waves: {point.ig_availability:.2f}%<br>"
                    f"Events/year: {point.frequency:.2f}<br>Mean duration: {point.duration:.1f} h<br>"
                    f"Valid hours: {int(point.observed_hours):,}/{int(point.expected_hours):,}")
            else:
                popup += (
                    f"Extraction: {escape(label)}<br>"
                    f"Months: {point.extracted_months}/{point.expected_months}<br>"
                    f"Valid historical hours: {int(point.valid_hours):,}/{int(point.cached_hours):,}<br>"
                    f"Retained raw months: {point.retained_months}/{point.extracted_months}<br>"
                    f"Source-gap hours: {int(point.source_gap_hours):,}<br>"
                    f"Analysis: {escape(point.analysis_status)}")
            folium.CircleMarker(
                [point.lat, point.lon], radius=3 if kind == "progress" else 5,
                color="#333333", weight=.4, fill=True, fill_color=colour, fill_opacity=.9,
                tooltip=f"Site {point.site_id}: {escape(label)}",
                popup=folium.Popup(popup, max_width=350),
            ).add_to(layer)
        layer.add_to(result)
    rows = "".join(
        f'<div><span style="display:inline-block;width:11px;height:11px;background:{colour};'
        f'margin-right:5px"></span>{escape(label)}: {int(points.map_class.eq(label).sum())}</div>'
        for label, colour in colours.items())
    thresholds = ("<div>Low &gt;95%; Medium &gt;85-95%;<br>High &gt;75-85%; Very high &lt;=75%</div>"
                  if kind == "availability" else "")
    result.get_root().html.add_child(folium.Element(
        '<div style="position:fixed;bottom:25px;left:15px;z-index:9999;background:white;'
        'padding:9px;border:1px solid #888;border-radius:4px;font-size:12px">'
        f'<details open><summary><b>{title}</b></summary>{rows}{thresholds}</details></div>'))
    folium.LayerControl(collapsed=True).add_to(result)
    return result


def read_results(root: Path, period: str, table: str = "summary",
                 method_signature: Optional[str] = None) -> pd.DataFrame:
    """Read verified stored products without needing raw files or the progress database."""
    if table not in {"summary", "annual", "seasonal"}:
        raise ValueError("Unknown result table.")
    config = json.loads((root / "workspace.json").read_text())
    if period not in config["periods"]:
        raise ValueError(f"Unknown period: {period}")
    frames = []
    for path in sorted((root / "products").glob("site-*.json.gz")):
        archive = read_archive(path)
        receipt = archive["periods"].get(period)
        if receipt is None or (method_signature is not None
                               and receipt["expected"]["method"] != method_signature):
            continue
        method, inputs = receipt["expected"]["method"], receipt["expected"]["inputs"]
        expected = expected_product(config, archive["site_id"], period, method, inputs)
        if verified_product(root, expected, archive=archive) is None:
            raise ValueError(f"Stored product belongs to another workspace/period: {path}")
        frames.append(pd.read_csv(io.StringIO(receipt["tables"][table])).assign(
            method_signature=method, source_inputs=inputs))
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)
    if result.method_signature.nunique() != 1:
        raise ValueError("Multiple stored methods present; select method_signature from available_periods().")
    return result
