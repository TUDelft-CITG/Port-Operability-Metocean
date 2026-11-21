"""Exploratory links between detected infrastructure and regional coastal exposure."""

from __future__ import annotations

import json
from pathlib import Path

import folium
import geopandas as gpd
import h3
import numpy as np
import pandas as pd
from pyproj import Geod
from scipy.spatial import cKDTree
from shapely.affinity import translate
from shapely.geometry import MultiPolygon, Polygon, box

from .analysis import RISK_COLOURS

RISK_ORDER = ["Low", "Medium", "High", "Very high"]
GEOD = Geod(ellps="WGS84")


def jetty_dimensions(jetties: pd.DataFrame) -> pd.DataFrame:
    """Geodesic OBB long/short axes: detection-size proxies, not surveyed lengths."""
    edges = []
    for corner in range(1, 5):
        following = corner % 4 + 1
        _, _, distance = GEOD.inv(
            jetties[f"corner_{corner}_x"].tolist(),
            jetties[f"corner_{corner}_y"].tolist(),
            jetties[f"corner_{following}_x"].tolist(),
            jetties[f"corner_{following}_y"].tolist())
        edges.append(np.asarray(distance))
    opposite = np.column_stack(((edges[0] + edges[2]) / 2, (edges[1] + edges[3]) / 2))
    if not np.isfinite(opposite).all() or (opposite <= 0).any():
        raise ValueError("Jetty boxes have missing, invalid or degenerate corner coordinates.")
    result = jetties.copy()
    result["obb_long_axis_m"] = opposite.max(axis=1)
    result["obb_short_axis_m"] = opposite.min(axis=1)
    return result


def load_infrastructure(root: Path, quality: str = "high_medium"):
    """Inventory memberships define unique structures; verified raw boxes supply size."""
    if quality not in {"high_medium", "all"}:
        raise ValueError("Infrastructure quality must be high_medium or all.")
    columns = ["cluster_id", "centroid_lon", "centroid_lat", "port_confidence",
               "assessment_status", "n_jetty", "n_members", "n_breakwater",
               "n_container", "n_liquid", "n_basin"]
    sites = pd.read_csv(root / "4.inventory/port_candidates_site_1km.csv", usecols=columns)
    original_count = len(sites)
    if sites.cluster_id.duplicated().any():
        raise ValueError("Candidate-site identifiers must be unique.")
    if quality == "high_medium":
        sites = sites.loc[sites.port_confidence.isin(["High", "Medium"])
                          & sites.assessment_status.eq("assessed")].copy()
    corners = [f"corner_{i}_{axis}" for i in range(1, 5) for axis in ("x", "y")]
    boxes = gpd.read_file(
        root / "1.raw/global_detections_raw.gpkg", layer="detection_obbs",
        where="class_name = 'Jetty'",
        columns=["raw_row_id", "confidence", "year", "center_x", "center_y", *corners])
    boxes = jetty_dimensions(pd.DataFrame(boxes.drop(columns="geometry")))
    members = pd.read_csv(root / "3.evidence/members_enriched.csv",
                          usecols=["det_id", "group_id", "cluster_1km", "class", "lon", "lat", "conf"])
    members = members.loc[members["class"].eq("Jetty")].copy()
    if members.det_id.duplicated().any() or members.group_id.duplicated().any():
        raise ValueError("Inventory detection memberships must be unique.")
    excluded = int((~members.cluster_1km.isin(sites.cluster_id)).sum())
    members = members.loc[members.cluster_1km.isin(sites.cluster_id)].copy()
    jetties = match_jetty_boxes(members, boxes)
    counts = jetties.groupby("cluster_1km").size()
    sites["jetty_count"] = sites.cluster_id.map(counts).fillna(0).astype(int)
    if not sites.jetty_count.eq(sites.n_jetty).all():
        raise ValueError("Linked deduplicated jetty counts disagree with inventory counts.")
    dimensions = jetties.groupby("cluster_1km").obb_long_axis_m
    sites["jetty_obb_total_m"] = sites.cluster_id.map(dimensions.sum()).fillna(0)
    sites["jetty_obb_median_m"] = sites.cluster_id.map(dimensions.median())
    return sites, jetties, {
        "inventory_sites": original_count, "selected_sites": len(sites),
        "selected_jetties": len(jetties),
        "jetties_excluded_by_site_selection": excluded,
        "box_source": "raw jetty OBBs; inventory memberships prevent duplicate counting",
        "box_matching": "same class, centre within 1 m, confidence within 0.0001",
        "maximum_box_match_distance_m": float(jetties.source_match_distance_m.max()),
        "length_interpretation": "OBB long-axis proxy, not surveyed centreline length",
    }


def match_jetty_boxes(members: pd.DataFrame, boxes: pd.DataFrame) -> pd.DataFrame:
    """IDs were reindexed between exports; verify spatial and confidence equivalence."""
    if boxes.empty or members.empty:
        raise ValueError("No jetty members or bounding boxes for this selection.")
    coordinates = np.column_stack((np.deg2rad(boxes.center_x), np.deg2rad(boxes.center_y)))
    reference = np.column_stack((np.deg2rad(members.lon), np.deg2rad(members.lat)))
    if (not np.isfinite(coordinates).all() or not np.isfinite(reference).all()
            or not np.isfinite(boxes.confidence).all() or not np.isfinite(members.conf).all()):
        raise ValueError("Missing source or inventory coordinates/confidence.")

    def sphere(values):
        x, y = values.T
        return np.column_stack((np.cos(y) * np.cos(x), np.cos(y) * np.sin(x), np.sin(y)))

    _, indices = cKDTree(sphere(coordinates)).query(sphere(reference), k=min(2, len(boxes)))
    if indices.ndim == 1:
        indices = indices[:, None]
    nearest = boxes.iloc[indices[:, 0]].reset_index(drop=True)
    _, _, distance = GEOD.inv(members.lon.tolist(), members.lat.tolist(),
                              nearest.center_x.tolist(), nearest.center_y.tolist())
    if ((np.asarray(distance) > 1).any()
            or not np.isfinite(distance).all()
            or np.abs(members.conf.to_numpy() - nearest.confidence.to_numpy()).max() > .0001):
        raise ValueError("Inventory jetty has no verified matching raw bounding box.")
    if indices.shape[1] == 2:
        second = boxes.iloc[indices[:, 1]]
        _, _, second_distance = GEOD.inv(
            members.lon.tolist(), members.lat.tolist(),
            second.center_x.tolist(), second.center_y.tolist())
        ambiguous = ((np.asarray(second_distance) <= 1)
                     & (np.abs(members.conf.to_numpy() - second.confidence.to_numpy()) <= .0001))
        disagreement = np.abs(second.obb_long_axis_m.to_numpy()
                              - nearest.obb_long_axis_m.to_numpy()) > .1
        if (ambiguous & disagreement).any():
            raise ValueError("Multiple matching source boxes have incompatible jetty size proxies.")
    result = members.reset_index(drop=True).copy()
    for field in ["raw_row_id", "year", "obb_long_axis_m", "obb_short_axis_m"]:
        result[field] = nearest[field].to_numpy()
    result["source_match_distance_m"] = np.asarray(distance)
    return result


def link_risk(locations: pd.DataFrame, sources: pd.DataFrame, summary: pd.DataFrame,
              lon: str, lat: str, max_distance_km: float = 75,
              min_coverage: float = 1.0) -> pd.DataFrame:
    """Nearest frozen ERA5 node, including missing nodes; never leapfrog missing results."""
    if (sources.empty or sources.site_id.duplicated().any()
            or summary.site_id.duplicated().any()
            or not np.isfinite(max_distance_km) or max_distance_km <= 0
            or not 0 <= min_coverage <= 1):
        raise ValueError("Select unique source sites and one period/scenario with valid limits.")

    def sphere(x, y):
        longitude, latitude = np.deg2rad(x), np.deg2rad(y)
        if (not np.isfinite(longitude).all() or not np.isfinite(latitude).all()
                or (np.abs(latitude) > np.pi / 2).any()):
            raise ValueError("Missing or invalid geographic coordinates.")
        return np.column_stack((np.cos(latitude) * np.cos(longitude),
                                np.cos(latitude) * np.sin(longitude), np.sin(latitude)))

    tree = cKDTree(sphere(sources.lon.to_numpy(), sources.lat.to_numpy()))
    _, nearest = tree.query(sphere(locations[lon].to_numpy(), locations[lat].to_numpy()))
    chosen = sources.iloc[nearest]
    _, _, distance = GEOD.inv(locations[lon].tolist(), locations[lat].tolist(),
                              chosen.lon.tolist(), chosen.lat.tolist())
    result = locations.copy()
    result["risk_site_id"] = chosen.site_id.to_numpy()
    result["risk_country"] = chosen.country.to_numpy()
    result["risk_distance_km"] = np.asarray(distance) / 1000
    fields = ["site_id", "risk", "total_availability", "main_driver", "coverage_fraction"]
    selected = summary[fields].rename(columns={"site_id": "risk_site_id"})
    result = result.merge(selected, on="risk_site_id", how="left", validate="many_to_one")
    result["link_status"] = "matched"
    result.loc[result.risk.isna(), "link_status"] = "no_result"
    result.loc[result.coverage_fraction.lt(min_coverage),
               "link_status"] = "insufficient_coverage"
    result.loc[result.risk_distance_km.gt(max_distance_km),
               "link_status"] = "beyond_distance_limit"
    unsupported = result.link_status.ne("matched")
    result.loc[unsupported, ["risk", "total_availability", "main_driver"]] = None
    return result


def hex_polygon(cell: str):
    """Split antimeridian cells so Leaflet never draws a world-spanning hexagon."""
    boundary = np.asarray(h3.cell_to_boundary(cell))
    longitude = np.rad2deg(np.unwrap(np.deg2rad(boundary[:, 1])))
    polygon = Polygon(np.column_stack((longitude, boundary[:, 0])))
    world = box(-180, -90, 180, 90)
    pieces = []
    for offset in (-360, 0, 360):
        clipped = translate(polygon, xoff=offset).intersection(world)
        if isinstance(clipped, Polygon) and not clipped.is_empty:
            pieces.append(clipped)
        elif isinstance(clipped, MultiPolygon):
            pieces.extend(clipped.geoms)
    if not pieces:
        raise ValueError(f"No display geometry for H3 cell {cell}.")
    return pieces[0] if len(pieces) == 1 else MultiPolygon(pieces)


def coastal_hexagons(sources: pd.DataFrame, summary: pd.DataFrame, resolution: int = 4,
                     max_distance_km: float = 75, min_coverage: float = 1.0):
    """One H3 neighbour ring around coastline anchors: a display band, not new observations."""
    if not 0 <= resolution <= 15:
        raise ValueError("H3 resolution must be between 0 and 15.")
    anchors = {h3.latlng_to_cell(p.anchor_lat, p.anchor_lon, resolution)
               for p in sources.itertuples()}
    cells = sorted({cell for anchor in anchors for cell in h3.grid_disk(anchor, 1)})
    centres = [h3.cell_to_latlng(cell) for cell in cells]
    locations = pd.DataFrame({"hex_id": cells, "lat": [p[0] for p in centres],
                              "lon": [p[1] for p in centres]})
    linked = link_risk(locations, sources, summary, "lon", "lat",
                       max_distance_km, min_coverage)
    return gpd.GeoDataFrame(linked, geometry=[hex_polygon(cell) for cell in cells], crs=4326)


def risk_comparison(sites: pd.DataFrame, jetties: pd.DataFrame) -> pd.DataFrame:
    """Site-normalised descriptive comparisons, including sites with zero detected jetties."""
    supported = sites.loc[sites.link_status.eq("matched")]
    group = supported.groupby("risk", observed=True)
    result = group.agg(candidate_sites=("cluster_id", "size"),
                       sites_with_jetties=("jetty_count", lambda x: int(x.gt(0).sum())),
                       detected_jetties=("jetty_count", "sum"),
                       mean_jetties_per_site=("jetty_count", "mean"),
                       median_jetties_per_site=("jetty_count", "median"),
                       total_jetty_obb_m=("jetty_obb_total_m", "sum"))
    result["share_sites_with_jetties"] = result.sites_with_jetties / result.candidate_sites
    linked = jetties.merge(supported[["cluster_id", "risk"]], left_on="cluster_1km",
                           right_on="cluster_id", how="inner", validate="many_to_one")
    lengths = linked.groupby("risk", observed=True).obb_long_axis_m
    result["median_jetty_obb_m"] = lengths.median()
    result["mean_jetty_obb_m"] = lengths.mean()
    return result.reindex([risk for risk in RISK_ORDER if risk in result.index])


def infrastructure_map(hexagons: gpd.GeoDataFrame, sites: pd.DataFrame,
                       land: gpd.GeoDataFrame, resolution: int = 4) -> folium.Map:
    """Translucent coastal zones and individual sites over switchable map backgrounds."""
    result = folium.Map(location=[15, 0], zoom_start=2, tiles="OpenStreetMap", prefer_canvas=True)
    result.default_js = [item for item in result.default_js if item[0] in {"leaflet", "jquery"}]
    result.default_css = [item for item in result.default_css if item[0] == "leaflet_css"]
    folium.Figure(height=600).add_child(result)
    result.get_root().header.add_child(folium.Element(
        "<style>.leaflet-container {background:#bfd2d9 !important;}</style>"))
    folium.TileLayer("Esri.WorldGrayCanvas", name="Light-grey map (Esri)", show=False).add_to(result)
    folium.GeoJson(
        json.loads(land[["ADMIN", "geometry"]].to_json()),
        name="Country/coast outlines (Natural Earth)", show=False,
        interactive=False, style_function=lambda feature: {
            "fill": False, "color": "#777", "opacity": .5, "weight": .5},
    ).add_to(result)
    colours = {**RISK_COLOURS, None: "#bbb"}
    folium.GeoJson(
        json.loads(hexagons.to_json()), name="Coastal risk zones (regional estimates)",
        style_function=lambda feature: {
            "fillColor": colours[feature["properties"]["risk"]],
            "fillOpacity": .4, "color": colours[feature["properties"]["risk"]], "weight": .2},
        tooltip=folium.GeoJsonTooltip(fields=["risk", "link_status"]),
        popup=folium.GeoJsonPopup(fields=["hex_id", "risk", "total_availability",
                                         "risk_site_id", "risk_distance_km", "link_status"]),
    ).add_to(result)
    fields = ["cluster_id", "port_confidence", "jetty_count", "jetty_obb_total_m",
              "risk", "risk_distance_km", "link_status"]
    detail = gpd.GeoDataFrame(
        sites[fields], geometry=gpd.points_from_xy(sites.centroid_lon, sites.centroid_lat),
        crs=4326)
    folium.GeoJson(
        json.loads(detail.to_json()), name="Individual candidate sites",
        marker=folium.CircleMarker(
            radius=2.5, color="#3b185e", weight=.5, fill=True, fill_opacity=.7),
        popup=folium.GeoJsonPopup(fields=fields),
    ).add_to(result)
    legend = "".join(f'<div><span style="color:{colour}">&#9632;</span> {risk}</div>'
                     for risk, colour in RISK_COLOURS.items())
    result.get_root().html.add_child(folium.Element(
        '<div style="position:fixed;bottom:25px;left:15px;z-index:9999;'
        'background:white;color:#222;padding:10px;border:1px solid #888">'
        '<b>Regional operability risk</b>' + legend +
        f'<div>Risk grid: H3 resolution {resolution}</div>' +
        '<div>Grey: unsupported / missing</div>'
        '<div>Purple: selected infrastructure</div></div>'))
    folium.LayerControl(collapsed=True).add_to(result)
    return result
