import tempfile
import json
import hashlib
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import xarray as xr
from shapely.geometry import box

import full_study as full
import port_operability as model


class FullStudyTests(unittest.TestCase):
    def setUp(self):
        self.sites = pd.DataFrame({
            "site_id": [1, 2], "lon": [18.0, 18.5], "lat": [-34.0, -34.0],
            "depth0": [100.0, 100.0], "shore_normal": [270.0, 270.0],
            "continent": ["Africa", "Africa"], "country": ["Example", "Example"],
        })

    def raw(self):
        shape = (744, 1, 2)
        values = {"u10": 2, "v10": 0, "swh": 1, "mwd": 270, "mwp": 8, "pp1d": 8}
        return xr.Dataset(
            {v: (("time", "latitude", "longitude"), np.full(shape, value, dtype=float))
             for v, value in values.items()},
            coords={"time": full._times(2018, 1), "latitude": [-34.0], "longitude": [18.0, 18.5]})

    def download_example(self, folder):
        with patch.object(model, "retrieve_era5") as retrieve, \
                patch.object(model, "open_era5", return_value=self.raw()):
            full.download_continents(self.sites, ["Africa"], [2018], folder, folder,
                                    request_budget=1, months=[1])
            self.assertEqual(retrieve.call_count, 1)

    def test_native_cells_keep_one_continent_across_antimeridian(self):
        countries = gpd.GeoDataFrame(
            {"ADMIN": ["West", "East"], "CONTINENT": ["Asia", "North America"]},
            geometry=[box(178, 0, 180, 2), box(-179, 0, -175, 2)], crs=4326)
        sites = gpd.GeoDataFrame(
            {"site_id": [1, 2], "anchor_lon": [-179.9, -176.0], "anchor_lat": [1.0, 1.0]},
            geometry=gpd.points_from_xy([-180, -176], [1, 1]), crs=4326)
        result = full.assign_continents(sites, countries)
        self.assertEqual(result.continent.tolist(), ["Asia", "North America"])

    def test_plan_covers_47_whole_years_and_only_selected_continent(self):
        with tempfile.TemporaryDirectory() as folder:
            report = full.download_estimate(
                self.sites, ["Africa"], list(range(1979, 2026)), Path(folder))
        self.assertEqual(report.months.iloc[0], 47 * 12)
        self.assertEqual(report.pending_requests_upper_bound.iloc[0], 47 * 12)
        self.assertEqual(report.continent.tolist(), ["Africa"])
        self.assertEqual(full._months([2025], [12]), [(2025, 12)])

    def test_budget_and_disk_rejected_before_network(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(model, "retrieve_era5") as get:
            root = Path(folder)
            with self.assertRaisesRegex(ValueError, "exceed budget"):
                full.download_continents(self.sites, ["Africa"], [2018], root, root, 0, months=[1])
            with patch.object(full.shutil, "disk_usage", return_value=type("Disk", (), {"free": 0})()):
                with self.assertRaises(OSError):
                    full.download_continents(self.sites, ["Africa"], [2018], root, root, 1, months=[1])
            get.assert_not_called()

    def test_monthly_pair_cache_is_complete_and_resumable(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.download_example(root)
            with xr.open_dataset(full.month_path(root, "Africa", 2018, 1)) as ds:
                full._check_month(ds, self.sites, 2018, 1)
                self.assertEqual(dict(ds.sizes), {"time": 744, "site": 2})
                self.assertTrue((ds.u10 == 2).all())
            with patch.object(model, "retrieve_era5") as get:
                full.download_continents(self.sites, ["Africa"], [2018], root, root, 0, months=[1])
                get.assert_not_called()
            with xr.open_dataset(full.month_path(root, "Africa", 2018, 1)) as ds:
                with self.assertRaisesRegex(ValueError, "timestamps"):
                    full._check_month(ds.isel(time=slice(1, None)), self.sites, 2018, 1)

    def test_analysis_reuses_checkpoints_and_marks_incomplete_annual_coverage(self):
        table = xr.Dataset({"ig_height": (("hs", "tp"), np.full((2, 2), .01))},
                           coords={"hs": [0, 14.6], "tp": [.1, 40]})
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.download_example(root)
            arguments = dict(
                sites=self.sites, continents=["Africa"], years=[2018], root=root,
                scenarios=[model.Scenario()], periods={"example": ("2018-01-01", "2018-02-01")},
                months=[1], batch_size=1)
            with patch.object(model, "ig_lookup", return_value=table):
                summary, annual, seasonal = full.analyse_continents(**arguments)
            self.assertEqual(summary.observed_hours.tolist(), [744, 744])
            self.assertEqual(summary.coverage_fraction.tolist(), [1, 1])
            self.assertFalse(annual.complete_year.any())
            self.assertEqual(set(seasonal.season), {"DJF"})
            with patch.object(model, "prepare_site") as prepare, patch.object(model, "ig_lookup") as lookup:
                again = full.analyse_continents(**arguments)[0]
                prepare.assert_not_called()
                lookup.assert_not_called()
                pd.testing.assert_frame_equal(summary, again)
            source = full.month_path(root, "Africa", 2018, 1).with_suffix(".nc.source.json")
            metadata = json.loads(source.read_text())
            metadata["sha256"] = "0" * 64
            model.write_json(source, metadata)
            with self.assertRaisesRegex(ValueError, "Monthly cache hash differs"):
                full.analyse_continents(**arguments)

    def test_expanded_ig_lookup_does_not_clip_global_storm_heights(self):
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(model, "bound_ig_unit", return_value=(.01, 100)):
            table = model.ig_lookup(Path(folder) / "ig.nc", 20, period_step=5,
                                    max_height=14.6, max_period=40)
            self.assertEqual(float(table.hs.max()), 14.6)
            self.assertEqual(float(table.tp.max()), 40)
            np.testing.assert_allclose(model.interpolate_ig(table, [13.5], [30]), [13.5**2 * .01])

    def test_raw_cleanup_removes_only_new_responses_after_month_cache_is_saved(self):
        for preexisting in [False, True]:
            with self.subTest(preexisting=preexisting), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                plan = model.download_plan(self.sites, [2018], [1])[0]
                raw = root / "raw" / plan["filename"]

                def retrieve(request, target):
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(b"test response")
                    model.write_json(target.with_suffix(".nc.request.json"),
                                     {"request": request, "sha256": model.file_hash(target)})

                if preexisting:
                    retrieve(plan["request"], raw)
                with patch.object(model, "retrieve_era5", side_effect=retrieve), \
                        patch.object(model, "open_era5", return_value=self.raw()):
                    full.download_continents(self.sites, ["Africa"], [2018], root, root,
                                             1, discard_new_raw=True, months=[1])
                self.assertEqual(raw.exists(), preexisting)
                self.assertTrue(full.month_path(root, "Africa", 2018, 1).exists())
                self.assertTrue(raw.with_suffix(".nc.request.json").exists())

    def test_cleanup_refuses_unknown_extraction_contents(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "response.nc"
            path.write_bytes(b"response")
            contents = path.with_suffix(".nc.contents")
            contents.mkdir()
            unknown = contents / "keep.txt"
            unknown.write_text("not a generated NetCDF member")
            with self.assertRaisesRegex(ValueError, "unexpected response contents"):
                full._discard_new_response(path)
            self.assertTrue(path.exists())
            self.assertTrue(unknown.exists())

    def grib_example(self):
        import eccodes

        messages = []
        for hour in range(2):
            for parameter in [165, 166, 140229, 140230, 140232, 140231]:
                handle = eccodes.codes_grib_new_from_samples("regular_ll_sfc_grib1")
                try:
                    for key, value in {
                        "Ni": 2, "Nj": 1, "latitudeOfFirstGridPointInDegrees": -34,
                        "latitudeOfLastGridPointInDegrees": -34,
                        "longitudeOfFirstGridPointInDegrees": 18,
                        "longitudeOfLastGridPointInDegrees": 18.5,
                        "iDirectionIncrementInDegrees": .5,
                        "paramId": parameter, "dataDate": 20180101, "dataTime": hour * 100,
                        "step": 0,
                    }.items():
                        eccodes.codes_set(handle, key, value)
                    eccodes.codes_set_values(handle, [parameter % 100, hour + 1])
                    messages.append(eccodes.codes_get_message(handle))
                finally:
                    eccodes.codes_release(handle)
        return messages

    def test_fragmented_grib_stream_keeps_paired_coordinates_and_all_hourly_fields(self):
        payload = b"".join(self.grib_example())
        times = pd.date_range("2018-01-01", periods=2, freq="h")
        chunks = (payload[i:i + 17] for i in range(0, len(payload), 17))
        ds, sha256, size = full._stream_grib_points(chunks, self.sites, times)
        self.assertEqual(size, len(payload))
        self.assertEqual(sha256, hashlib.sha256(payload).hexdigest())
        np.testing.assert_array_equal(ds.pp1d.values, [[31, 1], [31, 2]])
        np.testing.assert_array_equal(ds.site, self.sites.site_id)
        np.testing.assert_array_equal(ds.longitude, self.sites.lon)
        with self.assertRaisesRegex(ValueError, "exact GRIB grid nodes"):
            full._stream_grib_points([payload], self.sites.assign(lon=[18.25, 18.5]), times)

    def test_grib_stream_rejects_truncation_missing_fields_and_duplicates(self):
        messages = self.grib_example()
        times = pd.date_range("2018-01-01", periods=2, freq="h")
        for payload, error in [
            (b"".join(messages)[:-1], "Truncated"),
            (b"".join(messages[:-1]), "Missing GRIB hourly fields"),
            (b"".join(messages + messages[:1]), "Duplicate GRIB field"),
        ]:
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                full._stream_grib_points([payload], self.sites, times)

    def test_grib_transfer_retains_prefix_and_resumes_after_retry_limit(self):
        remote = MagicMock()
        remote.request_id = "example"
        remote.get_results.return_value.content_length = 6

        def interrupted(block):
            yield block
            raise requests.ConnectionError("Transfer interrupted")

        responses = []
        for offset, block in enumerate([b"a", b"b", b"c", b"def"]):
            response = MagicMock()
            response.__enter__.return_value = response
            response.status_code = 206
            response.headers = {"Content-Range": f"bytes {offset}-5/6"}
            response.iter_content.return_value = (interrupted(block) if offset < 3 else [block])
            responses.append(response)
        with tempfile.TemporaryDirectory() as folder, \
                patch("requests.get", side_effect=responses) as get, patch.object(full.time, "sleep"):
            directory = Path(folder)
            with self.assertRaisesRegex(RuntimeError, "3/6 bytes retained"):
                full._download_grib_file(remote, directory)
            partial = next(directory.glob("*.grib.part"))
            self.assertEqual(partial.read_bytes(), b"abc")
            result = full._download_grib_file(remote, directory)
            self.assertEqual(result.read_bytes(), b"abcdef")
            self.assertEqual([c.kwargs["headers"]["Range"] for c in get.call_args_list],
                             ["bytes=0-5", "bytes=1-5", "bytes=2-5", "bytes=3-5"])
            get.reset_mock()
            self.assertEqual(full._download_grib_file(remote, directory), result)
            get.assert_not_called()

    def test_grib_transfer_rejects_ignored_or_incorrect_ranges_without_appending(self):
        remote = MagicMock()
        remote.request_id = "example"
        remote.get_results.return_value.content_length = 6

        def interrupted():
            yield b"abc"
            raise requests.ConnectionError("Transfer interrupted")

        first = MagicMock()
        first.__enter__.return_value = first
        first.status_code = 206
        first.headers = {"Content-Range": "bytes 0-5/6"}
        first.iter_content.return_value = interrupted()
        with tempfile.TemporaryDirectory() as folder, patch("requests.get", return_value=first):
            directory = Path(folder)
            with self.assertRaisesRegex(RuntimeError, "3/6 bytes retained"):
                full._download_grib_file(remote, directory, attempts=1)
            partial = next(directory.glob("*.grib.part"))
            for status, headers in [(200, {}), (206, {"Content-Range": "bytes 0-2/6"})]:
                with self.subTest(status=status):
                    wrong = MagicMock()
                    wrong.__enter__.return_value = wrong
                    wrong.status_code, wrong.headers = status, headers
                    with patch("requests.get", return_value=wrong):
                        with self.assertRaisesRegex(ValueError, "honor the exact"):
                            full._download_grib_file(remote, directory)
                    wrong.iter_content.assert_not_called()
                    self.assertEqual(partial.read_bytes(), b"abc")

    def test_annual_grib_costing_errors_are_bounded_and_do_not_submit_jobs(self):
        for status, expected_calls in [(502, 3), (403, 1)]:
            with self.subTest(status=status), tempfile.TemporaryDirectory() as folder:
                response = requests.Response()
                response.status_code = status
                client = MagicMock()
                client.client.estimate_costs.side_effect = requests.HTTPError(response=response)
                with patch("cdsapi.Client", return_value=client), patch.object(full.time, "sleep"):
                    with self.assertRaises(requests.HTTPError):
                        full.download_grib_years(self.sites, ["Africa"], [2025], Path(folder), 1)
                self.assertEqual(client.client.estimate_costs.call_count, expected_calls)
                client.client.submit.assert_not_called()

    def test_annual_grib_request_saves_twelve_caches_and_resumes_without_network(self):
        times = pd.date_range("2025-01-01", "2026-01-01", freq="h", inclusive="left")
        yearly = xr.Dataset(
            {v: (("time", "site"), np.ones((len(times), 2), dtype="float32"))
             for v in full.VARIABLES},
            coords={"time": times, "site": self.sites.site_id.to_numpy(),
                    "longitude": ("site", self.sites.lon.to_numpy()),
                    "latitude": ("site", self.sites.lat.to_numpy())})
        client = MagicMock()
        temporary_error = requests.Response()
        temporary_error.status_code = 502
        client.client.estimate_costs.side_effect = [
            requests.HTTPError(response=temporary_error), {"cost": 1, "limit": 2}]
        remote = client.client.submit.return_value
        remote.request_id, remote.status = "example", "successful"
        client.client.get_remote.return_value = remote
        remote.get_results.return_value.content_length = 4
        response = MagicMock()
        response.ok = True
        response.__enter__.return_value = response
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch("cdsapi.Client", return_value=client) as factory:
                with self.assertRaisesRegex(ValueError, "exceed budget"):
                    full.download_grib_years(self.sites, ["Africa"], [2025], root, 0)
                factory.assert_not_called()
            with patch("cdsapi.Client", return_value=client), \
                    patch("requests.get", return_value=response), \
                    patch.object(full.time, "sleep"), \
                    patch.object(full, "_stream_grib_points", return_value=(yearly, "raw-hash", 4)):
                full.download_grib_years(self.sites, ["Africa"], [2025], root, 1)
            self.assertEqual(client.client.estimate_costs.call_count, 2)
            client.client.submit.assert_called_once()
            request = client.client.submit.call_args.args[1]
            self.assertEqual(request["data_format"], "grib")
            self.assertEqual(request["year"], ["2025"])
            self.assertEqual(len(request["month"]), 12)
            self.assertEqual(len(list((root / "monthly/africa").glob("*.nc"))), 12)
            with patch("cdsapi.Client") as factory:
                full.download_grib_years(self.sites, ["Africa"], [2025], root, 0)
                factory.assert_not_called()
            for path in (root / "monthly/africa").glob("*.nc"):
                path.unlink()
            temporary_grib = root / "temporary.grib"
            temporary_grib.write_bytes(b"GRIB")
            with patch("cdsapi.Client", return_value=client), \
                    patch.object(full, "_download_grib_file", return_value=temporary_grib) as transfer, \
                    patch.object(full, "_stream_grib_points", return_value=(yearly, "raw-hash", 4)):
                full.download_grib_years(self.sites, ["Africa"], [2025], root, 0,
                                         transfer_directory=root / "local")
            transfer.assert_called_once_with(remote, root / "local")
            self.assertFalse(temporary_grib.exists())
            self.assertEqual(len(list((root / "monthly/africa").glob("*.nc"))), 12)


if __name__ == "__main__":
    unittest.main()
