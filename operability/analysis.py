"""Loading and plotting existing results, with standalone map export."""

from __future__ import annotations

import io
import json
import logging
import time
from html import escape
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional

import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from branca.element import JavascriptLink
from matplotlib.lines import Line2D

import port_operability as model
from .workspace import Registry
from .products import expected_product, read_archive, verified_product

LOGGER = logging.getLogger("operability")
RISK_COLOURS = {"Low": "#00cc00", "Medium": "#ffff33",
                "High": "#ff7f50", "Very high": "#d73027"}
PROGRESS_COLOURS = {"missing": "#969696", "partial": "#e6b800",
                    "complete": "#238b8e", "complete with source gaps": "#d95f02"}
DRIVER_COLOURS = {"Wind": "#006400", "Short waves": "#ffb450",
                  "Short & IG waves": "#00bfff", "IG waves": "#c41e79"}
SEASON_ORDER = ["DJF", "MAM", "JJA", "SON"]


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


def prepare_seasonal_comparison(summary: pd.DataFrame, seasonal: pd.DataFrame,
                               scenario: str = "baseline", min_coverage: float = 1.0):
    """Use the same observed sites for the whole period and all four seasons."""
    if not np.isfinite(min_coverage) or not 0 <= min_coverage <= 1:
        raise ValueError("Minimum observation coverage must be between zero and one.")
    if summary.empty or seasonal.empty:
        LOGGER.warning("No stored summary/seasonal results for the comparison.")
        return summary.iloc[:0].copy(), seasonal.iloc[:0].copy()
    selected = summary.loc[summary.scenario.eq(scenario)].copy()
    seasons = seasonal.loc[seasonal.scenario.eq(scenario)].copy()
    if selected.empty or seasons.empty:
        LOGGER.warning("No stored summary/seasonal results for scenario %s.", scenario)
        return selected.iloc[:0], seasons.iloc[:0]
    if selected.site_id.duplicated().any() or seasons.duplicated(["site_id", "season"]).any():
        raise ValueError("Select unique sites and one row per site/season.")
    for column in ["period", "method_signature"]:
        if (selected[column].nunique() != 1 or seasons[column].nunique() != 1
                or selected[column].iloc[0] != seasons[column].iloc[0]):
            raise ValueError(f"Summary and seasonal results must share one {column}.")
    if not seasons.season.isin(SEASON_ORDER).all():
        raise ValueError("Unknown or missing seasonal label.")
    availability = ["total_availability", "wind_availability",
                    "short_wave_availability", "ig_availability"]
    if (not selected.coverage_fraction.between(0, 1).all()
            or not selected[availability].apply(lambda column: column.between(0, 100)).all().all()
            or not seasons[availability].apply(lambda column: column.between(0, 100)).all().all()
            or not np.isfinite(seasons[["frequency_per_season", "duration"]]).all().all()):
        raise ValueError("Missing or invalid availability, coverage or seasonal statistics.")
    complete = seasons.groupby("site_id").season.nunique().eq(len(SEASON_ORDER))
    eligible = complete.index[complete]
    initial_count = len(selected)
    selected = selected.loc[
        selected.coverage_fraction.ge(min_coverage) & selected.site_id.isin(eligible)].copy()
    if len(selected) < initial_count:
        LOGGER.warning(
            "Seasonal comparison excludes %d sites below coverage %.3f or missing a season.",
            initial_count - len(selected), min_coverage)
    seasons = seasons.loc[seasons.site_id.isin(selected.site_id)].copy()
    if selected.empty:
        return selected, seasons
    reference = selected[["site_id", "total_availability", "coverage_fraction"]].rename(
        columns={"total_availability": "overall_availability"})
    seasons = seasons.merge(reference, on="site_id", validate="many_to_one")
    seasons["availability_difference_pp"] = seasons.total_availability - seasons.overall_availability
    seasons["risk"] = model.risk_class(seasons.total_availability)
    seasons["main_driver"] = seasons.apply(model.main_driver, axis=1)
    return selected, seasons


def seasonal_statistics(summary: pd.DataFrame, seasonal: pd.DataFrame) -> pd.DataFrame:
    """Thesis Table 4.11-style equal-site means for a prepared, matched cohort."""
    if summary.empty or seasonal.empty:
        raise ValueError("No matched seasonal comparison results.")
    availability = ["total_availability", "wind_availability",
                    "short_wave_availability", "ig_availability"]
    result = seasonal.groupby("season")[availability + [
        "frequency_per_season", "duration", "availability_difference_pp"]].mean().reindex(SEASON_ORDER)
    result = result.rename(columns={"frequency_per_season": "events_per_year"})
    result["sites"] = seasonal.groupby("season").site_id.nunique().reindex(SEASON_ORDER)
    overall = summary[availability + ["frequency", "duration"]].mean().rename(
        {"frequency": "events_per_year"})
    overall["availability_difference_pp"] = 0.0
    overall["sites"] = len(summary)
    result.loc[summary.period.iloc[0]] = overall
    result["sites"] = result.sites.astype(int)
    result.index.name = "season_or_period"
    return result


def _world_axis(axis, countries: gpd.GeoDataFrame, title: str):
    countries.plot(ax=axis, facecolor="#eeefee", edgecolor="#888", linewidth=.3, zorder=0)
    axis.set(xlim=(-180, 180), ylim=(-90, 90), facecolor="#bfd2d9",
             xlabel="Longitude", ylabel="Latitude", title=title)
    axis.set_aspect("equal", adjustable="box")


def _availability_scatter(axis, points: pd.DataFrame, column: str):
    values = points[column]
    if not values.between(0, 100).all():
        raise ValueError("Availability maps require finite percentages between zero and 100.")
    colours = model.risk_class(values).map(RISK_COLOURS).astype(object)
    return axis.scatter(points.lon, points.lat, c=colours, s=10,
                        edgecolors="#333", linewidths=.2, zorder=2, rasterized=True)


def _availability_legend(figure):
    labels = ["Low risk: >95%", "Medium: >85-95%", "High: >75-85%", "Very high: <=75%"]
    handles = [Line2D([], [], linestyle="", marker="o", markerfacecolor=colour,
                      markeredgecolor="#333", label=label)
               for colour, label in zip(RISK_COLOURS.values(), labels)]
    figure.legend(handles=handles, loc="lower center", ncol=4, title="Conditional availability")


def plot_indicator_maps(summary: pd.DataFrame, countries: gpd.GeoDataFrame):
    """Four stored dominant-driver groups, using the paper's Figure 6 palette."""
    if summary.empty:
        raise ValueError("No calculated driver results.")
    if not summary.main_driver.isin(DRIVER_COLOURS).all():
        raise ValueError("Unknown or missing stored main-driver class.")
    figure, axes = plt.subplots(2, 2, figsize=(16, 10))
    drivers = ["Wind", "Short waves", "IG waves", "Short & IG waves"]
    for axis, driver in zip(axes.flat, drivers):
        points = summary.loc[summary.main_driver.eq(driver)]
        _world_axis(axis, countries, f"{driver}: {len(points):,} sites")
        axis.scatter(points.lon, points.lat, color=DRIVER_COLOURS[driver], s=10,
                     edgecolors="#333", linewidths=.2, zorder=2, rasterized=True)
    handles = [Line2D([], [], linestyle="", marker="o", markerfacecolor=DRIVER_COLOURS[driver],
                      markeredgecolor="#333", label=driver) for driver in drivers]
    figure.legend(handles=handles, loc="lower center", ncol=4,
                  title="Main driver (not overall operability risk)")
    figure.suptitle(f"Main causes of inoperability - {summary.period.iloc[0]}")
    figure.subplots_adjust(top=.92, bottom=.13, hspace=.25)
    return figure


def plot_seasonal_maps(seasonal: pd.DataFrame, countries: gpd.GeoDataFrame,
                       differences: bool = False):
    """Four seasonal risk maps or signed departures from the same sites' overall values."""
    if seasonal.empty:
        raise ValueError("No matched seasonal results.")
    figure, axes = plt.subplots(2, 2, figsize=(16, 10))
    if differences:
        values = seasonal.availability_difference_pp
        if not np.isfinite(values).all():
            raise ValueError("Seasonal differences require finite paired availabilities.")
        limit = max(1.0, float(values.abs().max()))
        normalise = plt.Normalize(vmin=-limit, vmax=limit)
    for axis, season in zip(axes.flat, SEASON_ORDER):
        points = seasonal.loc[seasonal.season.eq(season)]
        if points.empty:
            raise ValueError(f"No matched results for {season}.")
        _world_axis(axis, countries, season)
        if differences:
            scatter = axis.scatter(
                points.lon, points.lat, c=points.availability_difference_pp,
                cmap="RdYlGn", norm=normalise, s=10, edgecolors="#333", linewidths=.2,
                zorder=2, rasterized=True)
        else:
            _availability_scatter(axis, points, "total_availability")
    title = "Seasonal departure from whole-period availability" if differences else "Seasonal operability risk"
    figure.suptitle(f"{title} - {seasonal.period.iloc[0]}")
    figure.subplots_adjust(top=.92, bottom=.13, hspace=.25)
    if differences:
        figure.colorbar(scatter, ax=list(axes.flat), orientation="horizontal",
                        fraction=.04, pad=.1, label="Season minus whole period (percentage points)")
    else:
        _availability_legend(figure)
    return figure


def make_map(points: pd.DataFrame, countries: gpd.GeoDataFrame, kind: str = "progress",
             scenario: str = "baseline") -> folium.Map:
    """Return an embedded, tile-free progress or calculated-result map; never export HTML."""
    if kind not in {"progress", "availability", "driver"}:
        raise ValueError("Map kind must be progress, availability or driver.")
    points = points.copy()
    if kind in {"availability", "driver"}:
        points = points.loc[points.scenario.eq(scenario)]
        if kind == "driver":
            colours, title = DRIVER_COLOURS, "Main cause of inoperability"
            points["map_class"] = points.main_driver
            if not points.map_class.isin(colours).all():
                raise ValueError("Unknown or missing stored main-driver class.")
        else:
            colours, title = RISK_COLOURS, "Calculated operability risk"
            points["map_class"] = points.risk
    else:
        colours, title = PROGRESS_COLOURS, "Extraction progress"
        points["map_class"] = points.extraction_status
        points.loc[points.extraction_status.eq("complete") & points.source_gap_hours.gt(0),
                   "map_class"] = "complete with source gaps"
    if points.empty or points.site_id.duplicated().any():
        raise ValueError("Select one nonempty period/scenario with unique coastal points.")
    is_seasonal = "season" in points.columns
    if is_seasonal:
        if points.season.nunique() != 1 or not points.season.isin(SEASON_ORDER).all():
            raise ValueError("Select one season for an interactive map.")
        title += f" - {points.season.iloc[0]}"
    result = folium.Map(location=[15, 0], zoom_start=2, tiles=None, prefer_canvas=True)
    # Bootstrap's global styles are unnecessary for these circle-marker maps.
    result.default_js = [entry for entry in result.default_js if entry[0] in {"leaflet", "jquery"}]
    result.default_css = [entry for entry in result.default_css if entry[0] == "leaflet_css"]
    folium.Figure(height=600).add_child(result)
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
            if kind in {"availability", "driver"}:
                popup += (
                    f"Period: {escape(point.period)}<br>Scenario: {escape(scenario)}<br>"
                    f"Availability: {point.total_availability:.2f}%<br>Risk: {escape(str(point.risk))}<br>"
                    f"Main driver: {escape(point.main_driver)}<br>"
                    f"Wind: {point.wind_availability:.2f}%<br>"
                    f"Short waves: {point.short_wave_availability:.2f}%<br>"
                    f"IG waves: {point.ig_availability:.2f}%<br>")
                if is_seasonal:
                    popup += (
                        f"Season: {escape(point.season)}<br>"
                        f"Difference from whole period: {point.availability_difference_pp:+.2f} pp<br>"
                        f"Events/year in this season: {point.frequency_per_season:.2f}<br>"
                        f"Mean duration: {point.duration:.1f} h<br>"
                        f"Observed seasonal hours: {int(point.observed_hours):,}<br>"
                        f"Whole-period coverage: {point.coverage_fraction:.1%}")
                else:
                    popup += (
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
    if kind == "driver":
        thresholds = ("<div>Lowest individual conditional availability.</div>"
                      "<div>Short &amp; IG waves: gap &lt;1 percentage point unless wind is lower.</div>"
                      "<div>Driver availabilities overlap; these are not additive downtime shares.</div>")
    result.get_root().html.add_child(folium.Element(
        '<div style="position:fixed;bottom:25px;left:15px;z-index:9999;background:white;'
        'padding:9px;border:1px solid #888;border-radius:4px;font-size:12px">'
        f'<details open><summary><b>{title}</b></summary>{rows}{thresholds}</details></div>'))
    folium.LayerControl(collapsed=True).add_to(result)
    return result


def save_map_html(result: folium.Map, destination: Path) -> None:
    """Atomically replace a standalone export, retrying filesystem timeouts."""
    markup = result.get_root().render().encode("utf-8")
    for attempt in range(3):
        temporary = None
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(
                    dir=destination.parent, prefix=f".{destination.name}.",
                    suffix=".part", delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(markup)
            temporary.replace(destination)
            return
        except TimeoutError as error:
            if attempt == 2:
                raise
            LOGGER.warning("Map export timed out for %s (attempt %d/3); retrying: %s",
                           destination, attempt + 1, error)
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    LOGGER.warning("Could not remove temporary map export %s",
                                   temporary, exc_info=True)
        time.sleep(2 ** attempt)


def notebook_map_html(result: folium.Map) -> str:
    """Render without frames, keeping Leaflet independent of notebook AMD loaders."""
    root = result.get_root()
    root.render()
    leaflet_url = root.header._children["leaflet"].url.rsplit("/", 1)[0] + "/leaflet-src.esm.js"
    dependencies = [child.url for name, child in root.header._children.items()
                    if name != "leaflet" and isinstance(child, JavascriptLink)]
    header = "".join(child.render() for child in root.header._children.values()
                     if not isinstance(child, JavascriptLink))
    name = result.get_name()
    host = f"{name}_notebook"
    return f"""
{header}
<style>
#{host} {{position:relative;height:600px;transform:translateZ(0);overflow:hidden;}}
#{host} #{name} {{width:100% !important;height:100% !important;}}
</style>
<div id="{host}" data-operability-map="{name}">
{root.html.render()}
<div data-map-status role="status"
 style="position:absolute;top:10px;left:45px;z-index:10000;background:white;color:#222;padding:8px">
Loading interactive map...</div>
</div>
<script>
(async () => {{
    const host = document.getElementById({json.dumps(host)});
    const status = host.querySelector("[data-map-status]");
    try {{
        // Classic Leaflet can register with AMD without creating window.L.
        const modules = globalThis.__operabilityMapModules ||= {{}};
        const url = {json.dumps(leaflet_url)};
        const state = modules[url] ||= {{attempt: 0}};
        if (!state.promise) {{
            const source = new URL(url);
            if (state.attempt) source.searchParams.set("operability_retry", state.attempt);
            state.attempt++;
            state.promise = import(source.href).catch(error => {{
                delete state.promise;
                throw error;
            }});
        }}
        const L = await state.promise;
        const scripts = globalThis.__operabilityMapScripts ||= {{}};
        for (const url of {json.dumps(dependencies)}) {{
            if (!scripts[url]) {{
                scripts[url] = new Promise((resolve, reject) => {{
                    const script = document.createElement("script");
                    script.src = url;
                    script.onload = resolve;
                    script.onerror = () => {{
                        delete scripts[url];
                        reject(new Error("Could not load map dependency: " + url));
                    }};
                    document.head.appendChild(script);
                }});
            }}
            await scripts[url];
        }}
        {root.script.render()}
        {name}.invalidateSize();
        host._operabilityMap = {name};
        host.dataset.mapReady = "true";
        status.remove();
    }} catch (error) {{
        status.setAttribute("role", "alert");
        status.textContent = "Map could not render: " + error.message +
            ". Open the exported HTML map in a browser.";
        console.error(error);
    }}
}})();
</script>
"""


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
