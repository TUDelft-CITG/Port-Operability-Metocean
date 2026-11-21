import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from scipy.optimize import brentq
from shapely.geometry import LineString, box

import port_operability as model


class WaveTests(unittest.TestCase):
    def test_dispersion_against_independent_root_solver(self):
        periods = np.array([3.0, 7.0, 15.0, 25.0])
        for depth in [0.5, 15.0, 20.0, 1000.0]:
            expected = [brentq(lambda k: model.GRAVITY * k * np.tanh(k * depth)
                              - (2 * np.pi / period)**2, 1e-10, 10)
                        for period in periods]
            np.testing.assert_allclose(model.wave_number(periods, depth), expected, rtol=1e-8)

    def test_identity_depth_and_seaward_filter(self):
        result = model.nearshore_waves([1, 1], [8, 8], [0, 180], 20, 20, 0)
        np.testing.assert_allclose(result["short_wave"], [1, 0], atol=1e-12)
        np.testing.assert_allclose(result["all_direction_wave"], [1, 1], atol=1e-12)

    def test_breaking_cap(self):
        result = model.nearshore_waves([100], [10], [0], 100, 20, 0)
        self.assertAlmostEqual(result["short_wave"][0], 14.6)

    def test_bound_ig_quadrature_convergence(self):
        coarse = model.bound_ig_unit(10, n_frequency=128, n_direction=48)
        refined = model.bound_ig_unit(10, n_frequency=256, n_direction=96, n_ig=37)
        self.assertLess(abs(coarse[0] / refined[0] - 1), 0.03)
        self.assertGreater(coarse[1], 25)
        self.assertLess(coarse[1], 250)

    def test_invalid_wave_input_is_explicit(self):
        with self.assertRaises(ValueError):
            model.wave_number([0], 20)
        with self.assertRaises(ValueError):
            model.nearshore_waves([-1], [8], [0], 100, 20, 0)
        with self.assertRaises(ValueError):
            model.wave_number([8], np.nan)
        with self.assertRaises(ValueError):
            model.nearshore_waves([1], [8], [np.nan], 100, 20, 0)

    def test_ig_interpolation_and_domain_guard(self):
        table = xr.Dataset({"ig_height": (("hs", "tp"), [[0, 0], [0.01, 0.03]])},
                           coords={"hs": [0.0, 1.0], "tp": [5.0, 10.0]})
        np.testing.assert_allclose(model.interpolate_ig(table, [0.5], [7.5]), [0.01])
        with self.assertRaises(ValueError):
            model.interpolate_ig(table, [2], [7])


class EventTests(unittest.TestCase):
    def test_four_hour_gap_filled_but_five_hour_gap_retained(self):
        four = [False, True, False, False, False, False, True, False]
        five = [False, True, False, False, False, False, False, True, False]
        self.assertEqual(model.merge_short_gaps(four).tolist(),
                         [False, True, True, True, True, True, True, False])
        self.assertEqual(model.merge_short_gaps(five).tolist(), five)
        self.assertEqual(four, [False, True, False, False, False, False, True, False])

    def test_boundary_events_are_not_erased(self):
        index = pd.date_range("2018-01-01", periods=5, freq="h")
        storm = model.event_statistics([True] * 5, index)
        calm = model.event_statistics([False] * 5, index)
        self.assertEqual(storm["availability"], 0)
        self.assertEqual(storm["duration"], 5)
        self.assertEqual(storm["events"], 1)
        self.assertEqual(calm["availability"], 100)
        self.assertEqual(calm["events"], 0)

    def test_missing_hours_do_not_become_a_bridge(self):
        index = pd.to_datetime(["2018-01-01T00:00", "2018-01-01T01:00",
                                "2018-01-01T03:00", "2018-01-01T04:00"])
        frame = pd.DataFrame({"wind": [20, 0, 0, 20], "short_wave": 0.0, "ig": 0.0}, index=index)
        result = model.conditional_statistics(frame, model.Scenario())
        self.assertEqual(result["total_availability"], 50)
        self.assertEqual(result["duration"], 1)

    def test_joint_gap_created_by_different_indicators(self):
        index = pd.date_range("2018-01-01", periods=5, freq="h")
        frame = pd.DataFrame({"wind": [20, 0, 0, 0, 0],
                              "short_wave": [0, 0, 0, 0, 3], "ig": 0.0}, index=index)
        result = model.conditional_statistics(frame, model.Scenario())
        self.assertEqual(result["total_availability"], 0)
        self.assertEqual(result["raw_availability"], 60)

    def test_threshold_equality_is_unavailable(self):
        frame = pd.DataFrame({"wind": [13.8], "short_wave": [0.0], "ig": [0.0]},
                             index=pd.date_range("2018-01-01", periods=1))
        self.assertEqual(model.conditional_statistics(frame, model.Scenario())["total_availability"], 0)

    def test_risk_boundaries_match_paper(self):
        labels = list(model.risk_class(pd.Series([75, 75.1, 85, 85.1, 95, 95.1])).astype(str))
        self.assertEqual(labels, ["Very high", "High", "High", "Medium", "Medium", "Low"])

    def test_driver_joint_category_excludes_exact_one_percentage_point(self):
        row = pd.Series({"wind_availability": 100.0, "short_wave_availability": 90.0,
                         "ig_availability": 89.0})
        self.assertEqual(model.main_driver(row), "IG waves")
        row["short_wave_availability"] = 89.99
        self.assertEqual(model.main_driver(row), "Short & IG waves")

    def test_short_wave_percentiles_exclude_seaward_hours(self):
        frame = pd.DataFrame(
            {"wind": [0, 0], "short_wave": [0, 3], "ig": [0, 0], "shoreward": [False, True]},
            index=pd.date_range("2018-01-01", periods=2, freq="h"))
        result = model.conditional_statistics(frame, model.Scenario())
        self.assertEqual(result["short_wave_p50"], 3)
        self.assertEqual(result["short_wave_raw_availability"], 50)


class DataTests(unittest.TestCase):
    def test_download_budget_fails_before_network(self):
        sites = pd.DataFrame({"site_id": [1], "lon": [18.0], "lat": [-34.0]})
        with tempfile.TemporaryDirectory() as folder, patch("port_operability.retrieve_era5") as get:
            with self.assertRaisesRegex(ValueError, "exceed budget"):
                model.download_hourly_sites(sites, [2018], [1, 7], Path(folder), request_budget=1)
            get.assert_not_called()

    def test_cds_zip_streams_are_merged(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            coords = {"valid_time": pd.date_range("2018-01-01", periods=2, freq="h"),
                      "latitude": [-34.0], "longitude": [18.0]}
            files = []
            for variable in ["u10", "v10"]:
                path = root / f"{variable}.nc"
                xr.Dataset({variable: (("valid_time", "latitude", "longitude"),
                                       np.ones((2, 1, 1)))}, coords=coords).to_netcdf(path)
                files.append(path)
            archive = root / "response.download"
            with zipfile.ZipFile(archive, "w") as stream:
                for path in files:
                    stream.write(path, arcname=f"nested/{path.name}")
            result = model.open_era5(archive)
            self.assertEqual(set(result.data_vars), {"u10", "v10"})
            self.assertEqual(result.sizes["time"], 2)

    def test_geojson_longitude_latitude_order_and_properties(self):
        frame = pd.DataFrame([{"site_id": 1, "lon": 18.0, "lat": -34.0,
                               "total_availability": 50.0, "risk": "Very high"}])
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "points.geojson"
            model.export_points(frame, target, "Calculated test.")
            feature = json.loads(target.read_text())["features"][0]
            self.assertEqual(feature["geometry"]["coordinates"], [18.0, -34.0])
            self.assertEqual(feature["properties"]["risk"], "Very high")

    def test_monthly_plan_respects_cds_field_limit(self):
        sites = pd.DataFrame({"site_id": [1, 2], "lon": [18.0, 18.5], "lat": [-34.0, -34.5]})
        plans = model.download_plan(sites, [2018], [1, 7])
        self.assertEqual(len(plans), 2)
        self.assertTrue(all(len(item["request"]["month"]) == 1 for item in plans))

    def test_point_cache_scalar_coordinates_are_usable(self):
        index = pd.date_range("2018-01-01", periods=2, freq="h")
        raw = xr.Dataset(
            {name: ("time", values) for name, values in {
                "u10": [2, 3], "v10": [0, 0], "swh": [1, 1],
                "mwp": [8, 8], "pp1d": [8, 8], "mwd": [270, 270],
            }.items()},
            coords={"time": index, "longitude": 18.0, "latitude": -34.0},
        )
        table = xr.Dataset({"ig_height": (("hs", "tp"), [[0, 0], [0.01, 0.02]])},
                           coords={"hs": [0.0, 2.0], "tp": [5.0, 10.0]})
        site = pd.Series({"site_id": 1, "lon": 18.0, "lat": -34.0,
                          "depth0": 20.0, "shore_normal": 270.0})
        result = model.prepare_site(raw, site, table, model.Scenario())
        np.testing.assert_allclose(result.wind, [2, 3])
        self.assertEqual(len(result), 2)


class CoastalGridTests(unittest.TestCase):
    def candidates(self, longitudes):
        return pd.DataFrame({
            "site_id": np.arange(len(longitudes)), "lon": longitudes,
            "lat": 0.0, "depth0": 20.0, "distance_to_coast_deg": 0.25,
        })

    def test_one_regular_row_not_parallel_rows(self):
        first = self.candidates(np.arange(-2, 2.01, 0.5))
        second = first.assign(site_id=first.site_id + len(first), lat=-0.5,
                              distance_to_coast_deg=0.75)
        coast = gpd.GeoDataFrame(geometry=[LineString([(-2, 0.25), (2, 0.25)])], crs=4326)
        result = model.regularise_coastal_grid(
            pd.concat([first, second]), coast, first[["lon", "lat"]], simplify_degrees=0)
        np.testing.assert_allclose(result.lat, 0)
        np.testing.assert_allclose(np.diff(result.lon), 0.5)
        self.assertEqual(len(result), len(first))
        self.assertFalse(result.duplicated(["lon", "lat"]).any())

    def test_coverage_and_explicit_exclusions(self):
        candidates = self.candidates(np.arange(-4, 4.01, 0.5))
        reference = candidates.loc[candidates.lon <= 0, ["lon", "lat"]]
        coast = gpd.GeoDataFrame(geometry=[LineString([(-4, 0.25), (4, 0.25)])], crs=4326)
        exclusion = box(-3, -1, -2, 1)
        result = model.regularise_coastal_grid(
            candidates, coast, reference, [exclusion], coverage_radius_km=60)
        self.assertLessEqual(result.coverage_distance_km.max(), 60)
        self.assertLessEqual(result.lon.max(), 0.5)
        self.assertFalse(any(exclusion.buffer(0.8).covers(point)
                             for point in gpd.points_from_xy(result.lon, result.lat)))

    def test_assignment_does_not_skip_an_intervening_native_cell(self):
        candidates = pd.DataFrame({
            "site_id": [1, 2, 3], "lon": [-0.5, 0, 0], "lat": [0.5, 1, 1.5],
            "depth0": 20.0, "distance_to_coast_deg": [0.6, 0.1, 0.1],
        })
        coast = gpd.GeoDataFrame(
            geometry=[LineString([(0.1, 0.5), (0.1, 1.5)])], crs=4326)
        result = model.regularise_coastal_grid(candidates, coast, candidates[["lon", "lat"]])
        self.assertEqual(result.site_id.tolist(), [1, 2, 3])

    def test_antimeridian_is_not_a_gap_or_a_world_spanning_chord(self):
        candidates = self.candidates(np.array([179, 179.5, -180, -179.5, -179]))
        coast = gpd.GeoDataFrame(
            geometry=[LineString([(179, 0.25), (-179, 0.25)])], crs=4326)
        result = model.regularise_coastal_grid(candidates, coast, candidates[["lon", "lat"]])
        self.assertEqual(len(result), 5)
        self.assertTrue((result.lon.abs() >= 179).all())
        self.assertLess(result.snap_distance_km.max(), 30)

    def test_nearly_native_but_off_grid_coordinate_is_rejected(self):
        candidates = self.candidates(np.array([179.0, 179.5]))
        coast = gpd.GeoDataFrame(
            geometry=[LineString([(179, 0.25), (180, 0.25)])], crs=4326)
        candidates.loc[0, "lon"] += 0.0001
        with self.assertRaisesRegex(ValueError, "native 0.5-degree"):
            model.regularise_coastal_grid(candidates, coast, candidates[["lon", "lat"]])


if __name__ == "__main__":
    unittest.main()
