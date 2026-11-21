import unittest

import folium
import geopandas as gpd
import h3
import pandas as pd
from shapely.geometry import box

from operability.infrastructure import (
    coastal_hexagons, hex_polygon, infrastructure_map, jetty_dimensions, link_risk, match_jetty_boxes,
    risk_comparison,
)


class InfrastructureTests(unittest.TestCase):
    def sources(self):
        return pd.DataFrame({
            "site_id": [1, 2], "lon": [0., 1.], "lat": [0., 0.],
            "anchor_lon": [0., 1.], "anchor_lat": [0., 0.],
            "country": ["Example", "Example"],
        })

    def summary(self):
        return pd.DataFrame({
            "site_id": [2], "risk": ["High"], "total_availability": [80.],
            "main_driver": ["IG waves"], "coverage_fraction": [1.],
        })

    def test_nearest_missing_source_is_not_replaced_by_distant_available_source(self):
        positions = pd.DataFrame({"lon": [.01, 1., 5.], "lat": [0., 0., 0.]})
        result = link_risk(positions, self.sources(), self.summary(), "lon", "lat")
        self.assertEqual(result.link_status.tolist(), [
            "no_result", "matched", "beyond_distance_limit"])
        self.assertEqual(result.risk_site_id.tolist(), [1, 2, 2])
        self.assertTrue(pd.isna(result.risk.iloc[0]))
        incomplete = self.summary().assign(coverage_fraction=.8)
        result = link_risk(positions.iloc[[1]], self.sources(), incomplete, "lon", "lat")
        self.assertEqual(result.link_status.iloc[0], "insufficient_coverage")
        self.assertTrue(pd.isna(result.risk.iloc[0]))

    def test_antimeridian_links_and_hexagons_do_not_span_the_world(self):
        sources = self.sources().iloc[[0]].assign(lon=179.9, anchor_lon=179.9)
        summary = self.summary().assign(site_id=1)
        result = link_risk(pd.DataFrame({"lon": [-179.9], "lat": [0.]}),
                           sources, summary, "lon", "lat")
        self.assertEqual(result.link_status.iloc[0], "matched")
        self.assertLess(result.risk_distance_km.iloc[0], 25)
        geometry = hex_polygon(h3.latlng_to_cell(0, 179.99, 4))
        parts = geometry.geoms if geometry.geom_type == "MultiPolygon" else [geometry]
        self.assertTrue(geometry.is_valid)
        for part in parts:
            self.assertLess(part.bounds[2] - part.bounds[0], 2)

    def test_hex_band_keeps_missing_risk_and_valid_neighbour_cells(self):
        result = coastal_hexagons(self.sources(), self.summary())
        self.assertEqual(result.hex_id.nunique(), len(result))
        self.assertTrue(result.geometry.is_valid.all())
        self.assertTrue(result.loc[result.risk_site_id.eq(1), "risk"].isna().all())
        anchor = h3.latlng_to_cell(0, 0, 4)
        self.assertTrue(set(h3.grid_disk(anchor, 1)).issubset(set(result.hex_id)))

    def test_map_keeps_individual_sites_and_translucent_zones_over_basemaps(self):
        hexagons = coastal_hexagons(self.sources(), self.summary())
        sites = pd.DataFrame({
            "cluster_id": ["a"], "centroid_lon": [1.], "centroid_lat": [0.],
            "port_confidence": ["High"], "assessment_status": ["assessed"],
            "jetty_count": [2], "jetty_obb_total_m": [300.],
            "risk": ["High"], "risk_distance_km": [1.], "link_status": ["matched"],
        })
        land = gpd.GeoDataFrame({"ADMIN": ["Example"]}, geometry=[box(0, 0, 2, 2)], crs=4326)
        result = infrastructure_map(hexagons, sites, land)
        layers = list(result._children.values())
        tiles = [layer for layer in layers if isinstance(layer, folium.TileLayer)]
        self.assertEqual(len(tiles), 2)
        self.assertTrue(any("openstreetmap" in layer.tiles for layer in tiles))
        self.assertTrue(any("World_Light_Gray" in layer.tiles for layer in tiles))
        self.assertFalse(any("aggregat" in getattr(layer, "layer_name", "").lower()
                             for layer in layers))
        polygons = [layer for layer in layers if isinstance(layer, folium.GeoJson)]
        zones = next(layer for layer in polygons if layer.layer_name.startswith("Coastal risk"))
        opacity = zones.style_function(zones.data["features"][0])["fillOpacity"]
        self.assertGreater(opacity, 0)
        self.assertLessEqual(opacity, .5)
        outlines = next(layer for layer in polygons if layer.layer_name.startswith("Country/coast"))
        self.assertFalse(outlines.style_function(outlines.data["features"][0])["fill"])
        details = next(layer for layer in polygons if layer.layer_name.startswith("Candidate sites:"))
        self.assertTrue(details.show)
        self.assertEqual(details.data["features"][0]["geometry"]["coordinates"], [1., 0.])

    def test_map_groups_all_confidences_without_dropping_context_limited_candidates(self):
        sites = pd.DataFrame({
            "cluster_id": ["a", "b", "c", "d", "e"],
            "centroid_lon": [0., 1., 2., 3., 4.], "centroid_lat": [0.] * 5,
            "port_confidence": ["High", "Medium", "Low", "Likely natural", "Low"],
            "assessment_status": ["assessed"] * 4 + ["context_limited"],
            "jetty_count": [0] * 5, "jetty_obb_total_m": [0.] * 5,
            "risk": [None] * 5, "risk_distance_km": [1.] * 5,
            "link_status": ["no_result"] * 5,
        })
        land = gpd.GeoDataFrame({"ADMIN": ["Example"]}, geometry=[box(0, 0, 2, 2)], crs=4326)
        hexagons = coastal_hexagons(self.sources(), self.summary())
        result = infrastructure_map(hexagons, sites, land)
        groups = [layer for layer in result._children.values()
                  if isinstance(layer, folium.GeoJson)
                  and layer.layer_name.startswith("Candidate sites:")]
        self.assertEqual(len(groups), 4)
        plotted = []
        for layer in groups:
            self.assertTrue(layer.show)
            confidences = {f["properties"]["port_confidence"] for f in layer.data["features"]}
            self.assertEqual(len(confidences), 1)
            plotted.extend(layer.data["features"])
        self.assertCountEqual([f["properties"]["cluster_id"] for f in plotted], sites.cluster_id)
        context_limited = next(f for f in plotted if f["properties"]["cluster_id"] == "e")
        self.assertEqual(context_limited["properties"]["assessment_status"], "context_limited")
        self.assertEqual(context_limited["geometry"]["coordinates"], [4., 0.])
        control = next(layer for layer in result._children.values()
                       if isinstance(layer, folium.LayerControl))
        self.assertTrue(control.options["collapsed"])
        result.get_root().render()
        self.assertEqual(len([name for name in control.overlays if name.startswith("Candidate sites:")]), 4)
        with self.assertRaisesRegex(ValueError, "confidence"):
            infrastructure_map(hexagons, sites.assign(port_confidence=None), land)
        with self.assertRaisesRegex(ValueError, "confidence"):
            infrastructure_map(hexagons, sites.assign(port_confidence="Unknown"), land)

    def test_box_proxy_and_verified_matching_reject_incompatible_sources(self):
        boxes = pd.DataFrame({
            "corner_1_x": [0.], "corner_1_y": [0.],
            "corner_2_x": [.001], "corner_2_y": [0.],
            "corner_3_x": [.001], "corner_3_y": [.0001],
            "corner_4_x": [0.], "corner_4_y": [.0001],
            "center_x": [.0005], "center_y": [.00005],
            "confidence": [.8], "raw_row_id": [42], "year": [2024],
        })
        boxes = jetty_dimensions(boxes)
        self.assertAlmostEqual(boxes.obb_long_axis_m.iloc[0], 111.319, places=2)
        self.assertAlmostEqual(boxes.obb_short_axis_m.iloc[0], 11.057, places=2)
        member = pd.DataFrame({"lon": [.0005], "lat": [.00005], "conf": [.8]})
        matched = match_jetty_boxes(member, boxes)
        self.assertEqual(matched.raw_row_id.iloc[0], 42)
        with self.assertRaises(ValueError):
            match_jetty_boxes(member.assign(conf=.3), boxes)
        conflicting = pd.concat([boxes, boxes.assign(obb_long_axis_m=500)], ignore_index=True)
        with self.assertRaises(ValueError):
            match_jetty_boxes(member, conflicting)

    def test_risk_comparison_counts_zero_jetty_sites_without_double_counting(self):
        sites = pd.DataFrame({
            "cluster_id": ["a", "b", "c"], "risk": ["High", "High", None],
            "link_status": ["matched", "matched", "no_result"],
            "jetty_count": [2, 0, 1], "jetty_obb_total_m": [300, 0, 50],
        })
        jetties = pd.DataFrame({"cluster_1km": ["a", "a", "c"],
                                "obb_long_axis_m": [100, 200, 50]})
        result = risk_comparison(sites, jetties)
        self.assertEqual(result.loc["High", "candidate_sites"], 2)
        self.assertEqual(result.loc["High", "mean_jetties_per_site"], 1)
        self.assertEqual(result.loc["High", "median_jetty_obb_m"], 150)
        self.assertEqual(result.loc["High", "share_sites_with_jetties"], .5)


if __name__ == "__main__":
    unittest.main()
