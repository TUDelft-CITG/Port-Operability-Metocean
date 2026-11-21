import gzip
import json
import logging
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

import geopandas as gpd
import numpy as np
import pandas as pd
import pandas as pd
import requests
import xarray as xr
from shapely.geometry import box

import full_study as full
import process_global
from operability.analysis import (
    DRIVER_COLOURS, RISK_COLOURS, SEASON_ORDER, load_land, make_map, notebook_map_html,
    plot_availability, plot_indicator_maps, plot_seasonal_maps, prepare_seasonal_comparison,
    read_progress, read_results, save_map_html, seasonal_statistics,
)
from operability.processing import run
from operability.workspace import Registry, assign_regions, initialise, product_path


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root = self.base / "workspace"
        self.database = self.base / "local/progress.sqlite3"
        self.grid = self.base / "grid.geojson"
        self.countries = self.base / "countries.geojson"
        gpd.GeoDataFrame(
            {"site_id": [1, 2], "depth0": [100., 100.], "shore_normal": [270., 270.],
             "anchor_lon": [18., 18.5], "anchor_lat": [-34., -34.]},
            geometry=gpd.points_from_xy([18., 18.5], [-34., -34.]), crs=4326,
        ).to_file(self.grid, driver="GeoJSON")
        gpd.GeoDataFrame(
            {"ADMIN": ["Example"], "CONTINENT": ["Africa"]},
            geometry=[box(16, -36, 20, -32)], crs=4326,
        ).to_file(self.countries, driver="GeoJSON")

    def initialise(self, first=2025, last=2025):
        initialise(self.root, self.grid, self.countries, first, last, database=self.database)

    def year_data(self, registry, year, mask_second=False):
        sites = registry.sites
        times = pd.date_range(f"{year}-01-01", f"{year + 1}-01-01", freq="h", inclusive="left")
        constants = {"u10": 2, "v10": 0, "swh": .1, "mwd": 270, "mwp": 8, "pp1d": 8}
        ds = xr.Dataset(
            {name: (("time", "site"), np.full((len(times), 2), value, dtype="float32"))
             for name, value in constants.items()},
            coords={"time": times, "site": sites.site_id.to_numpy(),
                    "longitude": ("site", sites.lon.to_numpy()),
                    "latitude": ("site", sites.lat.to_numpy())})
        storm = (((times >= "2024-12-31T20") & (times <= "2024-12-31T22"))
                 | ((times >= "2025-01-01T02") & (times <= "2025-01-01T04")))
        ds.u10.values[storm] = 20
        if mask_second:
            ds.swh.values[:, 1] = np.nan
        return ds

    def save_year(self, registry, year, mask_second=False, months=range(1, 13)):
        ds = self.year_data(registry, year, mask_second)
        region = registry.sites.region.iloc[0]
        for month in months:
            path = full.month_path(registry.acquisition_root(region), "Africa", year, month)
            full._save_month(ds.sel(time=full._times(year, month)), registry.sites,
                             year, month, path, [])

    def test_initialise_is_fresh_and_immutable_without_network(self):
        with patch.object(full, "download_grib_years") as download:
            self.initialise()
            download.assert_not_called()
        with Registry(self.root) as registry:
            points = registry.progress()
            self.assertEqual(points.extraction_status.tolist(), ["missing", "missing"])
            self.assertEqual(points.expected_months.tolist(), [12, 12])
            self.assertEqual(registry.db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0], 1)
            self.assertFalse(registry.config["legacy_metocean_data_imported"])
        self.assertEqual(list((self.root / "downloads").iterdir()), [])
        with self.assertRaises(FileExistsError):
            self.initialise()

    def test_progress_map_embeds_with_nonzero_iframe_height(self):
        self.initialise()
        with Registry(self.root) as registry:
            result = make_map(registry.progress(), load_land(self.root))
        markup = result._repr_html_()
        self.assertRegex(markup, r'^<iframe\b[^>]* height="[1-9][0-9]*"')
        self.assertNotIn("padding-bottom:", markup)
        self.assertEqual(result.get_root().render().count("L.circleMarker("), 2)

    def comparison_data(self):
        summary = pd.DataFrame({
            "site_id": [1, 2, 3], "lon": [18., 18.5, 19.], "lat": [-34.] * 3,
            "country": ["Example"] * 3, "scenario": ["baseline"] * 3,
            "period": ["year_2025"] * 3, "method_signature": ["one-method"] * 3,
            "coverage_fraction": [1., .5, 1.], "total_availability": [95., 80., 90.],
            "wind_availability": [99.] * 3, "short_wave_availability": [95.] * 3,
            "ig_availability": [95.5] * 3, "main_driver": ["Short & IG waves"] * 3,
            "risk": ["Medium", "High", "Medium"], "frequency": [7., 8., 9.],
            "duration": [12.] * 3, "observed_hours": [8760, 4380, 8760],
            "expected_hours": [8760] * 3,
        })
        rows = []
        for point in summary.to_dict("records"):
            for season, availability in zip(SEASON_ORDER, [90., 95., 100., 95.]):
                if point["site_id"] == 3 and season == "SON":
                    continue
                rows.append({
                    **{key: point[key] for key in [
                        "site_id", "lon", "lat", "country", "scenario", "period", "method_signature"]},
                    "season": season, "total_availability": availability,
                    "wind_availability": 99., "short_wave_availability": 95.,
                    "ig_availability": 95.5, "frequency": 80.,
                    "frequency_per_season": 2., "duration": 10., "observed_hours": 2160,
                })
        return summary, pd.DataFrame(rows)

    def test_seasonal_comparison_pairs_sites_and_uses_calendar_season_frequency(self):
        summary, seasons = self.comparison_data()
        original = seasons.copy()
        with self.assertLogs("operability", level="WARNING"):
            overall, matched = prepare_seasonal_comparison(summary, seasons)
        self.assertEqual(overall.site_id.tolist(), [1])
        self.assertEqual(matched.site_id.tolist(), [1] * 4)
        self.assertEqual(matched.availability_difference_pp.tolist(), [-5., 0., 5., 0.])
        self.assertEqual(matched.main_driver.tolist(), ["Short & IG waves"] * 4)
        table = seasonal_statistics(overall, matched)
        self.assertEqual(table.loc["DJF", "events_per_year"], 2.)
        self.assertEqual(table.loc["year_2025", "events_per_year"], 7.)
        self.assertEqual(table.loc["DJF", "availability_difference_pp"], -5.)
        self.assertEqual(table.sites.tolist(), [1] * 5)
        pd.testing.assert_frame_equal(seasons, original)
        with self.assertLogs("operability", level="WARNING"):
            relaxed, _ = prepare_seasonal_comparison(summary, seasons, min_coverage=.5)
        self.assertEqual(relaxed.site_id.tolist(), [1, 2])

    def test_seasonal_comparison_rejects_ambiguous_or_mixed_results(self):
        summary, seasons = self.comparison_data()
        for invalid in [pd.concat([seasons, seasons.iloc[[0]]]),
                        seasons.assign(period="year_2024"),
                        seasons.assign(method_signature="another-method"),
                        seasons.assign(season="unknown")]:
            with self.subTest(invalid=invalid.iloc[0].to_dict()), self.assertRaises(ValueError):
                prepare_seasonal_comparison(summary, invalid)

    def test_driver_and_season_maps_keep_risk_and_frequency_popup_meanings(self):
        import folium

        summary, seasons = self.comparison_data()
        land = gpd.read_file(self.countries)
        points = pd.concat([summary.iloc[[0]].assign(site_id=i + 10, main_driver=driver)
                            for i, driver in enumerate(DRIVER_COLOURS)], ignore_index=True)
        result = make_map(points, land, kind="driver")
        groups = [layer for layer in result._children.values()
                  if isinstance(layer, folium.FeatureGroup)]
        self.assertEqual([layer.layer_name for layer in groups], list(DRIVER_COLOURS))
        for group, colour in zip(groups, DRIVER_COLOURS.values()):
            marker = next(iter(group._children.values()))
            self.assertEqual(marker.options["fillColor"], colour)
        markup = result.get_root().render()
        self.assertEqual(markup.count("L.circleMarker("), 4)
        self.assertNotIn("Risk: Wind", markup)
        self.assertIn("Risk: Medium", markup)
        with self.assertLogs("operability", level="WARNING"):
            _, matched = prepare_seasonal_comparison(summary, seasons)
        markup = make_map(matched.loc[matched.season.eq("DJF")], land,
                          kind="availability").get_root().render()
        self.assertIn("Season: DJF", markup)
        self.assertIn("Events/year in this season: 2.00", markup)
        self.assertIn("Observed seasonal hours: 2,160", markup)
        self.assertIn("Difference from whole period: -5.00 pp", markup)
        with self.assertRaises(ValueError):
            make_map(matched, land, kind="availability")

    def test_seasonal_figures_preserve_class_boundaries_and_signed_shared_scale(self):
        from matplotlib.colors import to_hex
        import matplotlib.pyplot as plt

        summary, seasons = self.comparison_data()
        seasons = seasons.loc[seasons.site_id.eq(1)].copy()
        seasons["total_availability"] = [75., 85., 95., 100.]
        overall, matched = prepare_seasonal_comparison(summary.iloc[[0]], seasons)
        land = gpd.read_file(self.countries)
        figures = [plot_indicator_maps(overall, land), plot_seasonal_maps(matched, land),
                   plot_seasonal_maps(matched, land, differences=True)]
        for figure in figures:
            self.addCleanup(plt.close, figure)
        self.assertEqual(len(figures[0].axes), 4)
        self.assertEqual(len(figures[1].axes), 4)
        for axis, risk in zip(figures[1].axes, ["Very high", "High", "Medium", "Low"]):
            self.assertEqual(to_hex(axis.collections[-1].get_facecolors()[0]), RISK_COLOURS[risk])
        for axis, difference in zip(figures[2].axes[:4], [-20., -10., 0., 5.]):
            scatter = axis.collections[-1]
            self.assertEqual(scatter.get_array().tolist(), [difference])
            self.assertEqual(scatter.get_clim(), (-20., 20.))
            np.testing.assert_allclose(scatter.get_offsets(), [[18., -34.]])

    def test_driver_panels_partition_stored_classes_in_two_by_two_layout(self):
        from matplotlib.colors import to_hex
        import matplotlib.pyplot as plt

        summary, _ = self.comparison_data()
        drivers = ["Wind", "Short waves", "IG waves", "Short & IG waves"]
        points = pd.concat([
            summary.iloc[[0]].assign(site_id=i + 10, lon=18. + i, main_driver=driver)
            for i, driver in enumerate(drivers)], ignore_index=True)
        figure = plot_indicator_maps(points, gpd.read_file(self.countries))
        self.addCleanup(plt.close, figure)
        self.assertEqual(len(figure.axes), 4)
        for i, (axis, driver) in enumerate(zip(figure.axes, drivers)):
            layout = axis.get_subplotspec().get_gridspec()
            self.assertEqual((layout.nrows, layout.ncols), (2, 2))
            scatter = axis.collections[-1]
            np.testing.assert_allclose(scatter.get_offsets(), [[18. + i, -34.]])
            self.assertEqual(to_hex(scatter.get_facecolors()[0]), DRIVER_COLOURS[driver])
            self.assertEqual(axis.get_title(), f"{driver}: 1 sites")
        with self.assertRaisesRegex(ValueError, "main-driver"):
            plot_indicator_maps(points.assign(main_driver="unknown"), gpd.read_file(self.countries))

    def test_map_export_retries_partial_write_and_closes_files(self):
        destination = self.base / "maps/progress.html"
        destination.parent.mkdir()
        destination.write_text("previous map")
        result = MagicMock()
        result.get_root.return_value.render.return_value = "<html>new map</html>"
        handles = []
        create = tempfile.NamedTemporaryFile

        def interrupted_file(*args, **kwargs):
            self.assertEqual(destination.read_text(), "previous map")
            handle = create(*args, **kwargs)
            handles.append(handle)
            if len(handles) == 1:
                write = handle.write

                def interrupted_write(data):
                    write(data[:5])
                    raise TimeoutError("Cloud write timed out")

                handle.write = interrupted_write
            return handle

        with patch("operability.analysis.NamedTemporaryFile", side_effect=interrupted_file), \
                patch("operability.analysis.time.sleep") as sleep, \
                self.assertLogs("operability", level="WARNING") as logs:
            save_map_html(result, destination)
        self.assertEqual(destination.read_text(), "<html>new map</html>")
        self.assertEqual(len(handles), 2)
        self.assertTrue(all(handle.file.closed for handle in handles))
        self.assertEqual(list(destination.parent.iterdir()), [destination])
        sleep.assert_called_once_with(1)
        self.assertIn("retrying", logs.output[0])

    def test_map_export_preserves_previous_html_and_propagates_failures(self):
        destination = self.base / "progress.html"
        result = MagicMock()
        result.get_root.return_value.render.return_value = "<html>new map</html>"
        for error, attempts in [(TimeoutError("Cloud rename timed out"), 3),
                                (PermissionError("Read-only directory"), 1)]:
            with self.subTest(error=type(error).__name__):
                destination.write_text("previous map")
                with patch.object(Path, "replace", side_effect=error) as replace, \
                        patch("operability.analysis.time.sleep") as sleep:
                    with self.assertRaises(type(error)):
                        save_map_html(result, destination)
                self.assertEqual(replace.call_count, attempts)
                self.assertEqual(sleep.call_count, attempts - 1)
                self.assertEqual(destination.read_text(), "previous map")
                self.assertEqual(list(self.base.glob(".progress.html.*.part")), [])

    def test_notebook_progress_cell_refreshes_data_and_overwrites_html(self):
        from IPython.display import HTML, FileLink

        self.initialise()
        notebook = json.loads(Path(__file__).with_name("analyse-global.ipynb").read_text())
        source = next("".join(cell["source"]) for cell in notebook["cells"]
                      if cell["cell_type"] == "code" and "PROGRESS_HTML =" in "".join(cell["source"]))
        output = MagicMock()
        namespace = {
            "WORKSPACE": self.root, "COUNTRIES": [], "read_progress": read_progress,
            "make_map": make_map, "notebook_map_html": notebook_map_html,
            "save_map_html": save_map_html,
            "HTML": HTML, "FileLink": FileLink, "display": output,
            "land": load_land(self.root),
        }
        destination = self.root / "maps/progress.html"
        destination.parent.mkdir()
        destination.write_text("obsolete")
        exec(compile(source, "analyse-global.ipynb:progress-map", "exec"), namespace)
        first = destination.read_text()
        self.assertNotIn("obsolete", first)
        self.assertIn("missing: 2", first)
        self.assertEqual(first.count("L.circleMarker("), 2)
        inline = output.call_args_list[0].args[0].data
        self.assertNotIn("<iframe", inline)
        self.assertIn("data-operability-map=", inline)
        with Registry(self.root) as registry:
            self.save_year(registry, 2025, months=[1])
            registry.reconcile()
        exec(compile(source, "analyse-global.ipynb:progress-map", "exec"), namespace)
        self.assertEqual(namespace["progress"].extracted_months.tolist(), [1, 1])
        self.assertIn("partial: 2", destination.read_text())
        output.reset_mock()
        namespace["save_map_html"] = MagicMock(side_effect=TimeoutError("Export timed out"))
        with self.assertRaises(TimeoutError):
            exec(compile(source, "analyse-global.ipynb:progress-map", "exec"), namespace)
        self.assertEqual(output.call_count, 1)
        self.assertIsInstance(output.call_args.args[0], HTML)

    def test_cli_keeps_session_log_outside_workspace_and_preserves_failure(self):
        self.initialise()
        handlers = []

        def failed_run(registry, *args):
            handler = next(h for h in process_global.LOGGER.handlers
                           if isinstance(h, logging.FileHandler))
            self.assertEqual(Path(handler.baseFilename).parent,
                             self.database.parent.resolve() / "logs")
            handlers.append(handler)
            process_global.LOGGER.warning("Acquisition failed; request receipt retained.")
            return 1

        with patch.object(process_global, "run", side_effect=failed_run):
            self.assertEqual(process_global.main(
                ["--workspace", str(self.root), "run", "--hours", "0"]), 1)
        self.assertIsNone(handlers[0].stream)
        self.assertNotIn(handlers[0], process_global.LOGGER.handlers)
        self.assertIn("request receipt retained", Path(handlers[0].baseFilename).read_text())
        self.assertEqual(list((self.root / "logs").iterdir()), [])

    def test_dateline_regions_avoid_near_global_bounding_boxes(self):
        sites = pd.DataFrame({"site_id": [1, 2, 3, 4], "continent": ["Oceania"] * 4,
                              "lon": [-179, -175, 172, 179], "lat": [-10] * 4})
        grouped = assign_regions(sites)
        self.assertEqual(grouped.region.nunique(), 2)
        self.assertTrue(all(g.lon.max() - g.lon.min() < 10
                            for _, g in grouped.groupby("region")))

    def test_partial_coverage_source_gaps_and_removed_files_are_distinct(self):
        self.initialise()
        with Registry(self.root) as registry:
            self.save_year(registry, 2025, mask_second=True, months=[1])
            registry.reconcile()
            points = registry.progress()
            self.assertEqual(points.extraction_status.tolist(), ["partial", "partial"])
            self.assertEqual(points.valid_hours.tolist(), [744, 0])
            self.assertEqual(points.source_gap_hours.tolist(), [0, 744])
            self.save_year(registry, 2025, mask_second=True)
            registry.reconcile()
            points = registry.progress()
            self.assertEqual(points.extraction_status.tolist(), ["complete", "complete"])
            self.assertEqual(points.analysis_status.tolist(), ["ready", "blocked"])
            path = full.month_path(registry.acquisition_root(registry.sites.region.iloc[0]),
                                   "Africa", 2025, 1)
            path.unlink()
            registry.reconcile()
            self.assertEqual(registry.progress().extracted_months.tolist(), [12, 12])
            self.assertEqual(registry.progress().retained_months.tolist(), [11, 11])
            self.assertEqual(registry.progress().analysis_status.tolist(), ["needs_raw", "blocked"])

    def test_timeout_preserves_job_and_resume_needs_no_new_request_budget(self):
        self.initialise()
        with Registry(self.root) as registry:
            region = registry.sites.region.iloc[0]
            manifest = registry.request_path(region, 2025)

            def waiting(*args, **kwargs):
                manifest.parent.mkdir(parents=True, exist_ok=True)
                manifest.write_text(json.dumps({"request_id": "existing-job", "status": "running"}))
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    row = registry.db.execute("SELECT status,request_id FROM jobs").fetchone()
                    if tuple(row) == ("submitted", "existing-job"):
                        break
                    time.sleep(.05)
                self.assertEqual(tuple(row), ("submitted", "existing-job"))
                raise full.CDSWaitTimeout("Still queued")

            def complete(*args, **kwargs):
                self.assertEqual(json.loads(manifest.read_text())["request_id"], "existing-job")
                self.assertEqual(kwargs["request_budget"], 0)
                self.assertEqual(kwargs["transfer_directory"], registry.database.parent / "transfers")
                self.save_year(registry, 2025)
                manifest.write_text(json.dumps({"request_id": "existing-job", "status": "successful"}))

            with patch.object(full, "download_grib_years", side_effect=waiting):
                self.assertEqual(run(registry, hours=.02, stage="download",
                                     dashboard_address="127.0.0.1:0"), 0)
            self.assertEqual(registry.db.execute("SELECT status FROM jobs").fetchone()[0], "submitted")
            with patch.object(full, "download_grib_years", side_effect=complete) as download:
                self.assertEqual(run(registry, hours=.02, max_requests=0, stage="download",
                                     dashboard_address="127.0.0.1:0"), 0)
                download.assert_called_once()
            with patch.object(full, "download_grib_years") as download:
                self.assertEqual(run(registry, hours=0), 0)
                download.assert_not_called()

    def test_filesystem_timeout_is_not_reported_as_expired_cds_wait(self):
        self.initialise()
        with Registry(self.root) as registry, \
                patch.object(full, "download_grib_years",
                             side_effect=TimeoutError(60, "Operation timed out")), \
                self.assertLogs("operability", level="ERROR") as messages:
            self.assertEqual(run(registry, hours=.02, stage="download",
                                 dashboard_address="127.0.0.1:0"), 1)
            self.assertEqual(registry.db.execute("SELECT status FROM jobs").fetchone()[0], "failed")
            self.assertIn("Acquisition failed", "\n".join(messages.output))

    def test_corrupt_cache_is_not_complete_and_retry_quarantines_only_bad_files(self):
        self.initialise()
        with Registry(self.root) as registry:
            self.save_year(registry, 2025)
            registry.reconcile()
            path = full.month_path(registry.acquisition_root(registry.sites.region.iloc[0]),
                                   "Africa", 2025, 1)
            path.write_bytes(b"corrupt")
            self.assertEqual(registry.reconcile(), 1)
            self.assertEqual(registry.db.execute("SELECT status FROM jobs").fetchone()[0], "failed")
            self.assertEqual(registry.progress().retained_months.tolist(), [11, 11])
            self.assertEqual(registry.reconcile(repair=True), 0)
            self.assertFalse(path.exists())
            self.assertEqual(len(list((self.root / "quarantine").iterdir())), 2)

    def test_scoped_retry_does_not_modify_other_years(self):
        self.initialise(2024, 2025)
        with Registry(self.root) as registry:
            self.save_year(registry, 2024)
            registry.reconcile()
            path = full.month_path(registry.acquisition_root(registry.sites.region.iloc[0]),
                                   "Africa", 2024, 1)
            path.write_bytes(b"corrupt")
            self.assertEqual(registry.reconcile(years=[2025], repair=True), 0)
            self.assertEqual(path.read_bytes(), b"corrupt")
            self.assertEqual(list((self.root / "quarantine").iterdir()), [])
            self.assertEqual(registry.reconcile(), 1)

    def test_two_dask_workers_dashboard_and_continuous_history_survive_database_rebuild(self):
        from distributed import LocalCluster

        self.initialise(2024, 2025)
        dashboards = []

        def checked_cluster(**kwargs):
            cluster = LocalCluster(**kwargs)
            url = cluster.dashboard_link
            response = requests.get(url, timeout=10)
            self.assertEqual(response.status_code, 200)
            counts = requests.get(url.replace("/status", "/json/counts.json"), timeout=10).json()
            self.assertEqual(counts["workers"], 2)
            dashboards.append(url)
            return cluster

        with Registry(self.root) as registry:
            for year in [2024, 2025]:
                self.save_year(registry, year)
            with patch("distributed.LocalCluster", side_effect=checked_cluster):
                self.assertEqual(run(registry, hours=.1, workers=2, batch_size=1,
                                     stage="calculate", dashboard_address="127.0.0.1:0"), 0)
            self.assertEqual(registry.progress().analysis_status.tolist(), ["complete", "complete"])
            self.assertEqual(registry.db.execute(
                "SELECT COUNT(*) FROM analyses WHERE status='complete'").fetchone()[0], 6)
            self.assertEqual(len(list((self.root / "products").glob("site-*.json.gz"))), 2)
        self.assertEqual(len(dashboards), 1)
        history = read_results(self.root, "extended_2024_2025")
        baseline = history.loc[history.scenario.eq("baseline")]
        self.assertEqual(baseline.events.tolist(), [1, 1])
        self.assertEqual(baseline.duration.tolist(), [9, 9])
        map_html = make_map(history, load_land(self.root), kind="availability").get_root().render()
        self.assertEqual(map_html.count("L.circleMarker("), 2)
        self.assertIn("<details open>", map_html)
        self.assertNotIn("L.tileLayer(", map_html)
        import matplotlib.pyplot as plt
        figure = plot_availability(history, read_results(self.root, "extended_2024_2025", "seasonal"))
        plt.close(figure)
        for year in [2024, 2025]:
            annual = read_results(self.root, f"year_{year}")
            self.assertEqual(annual.loc[annual.scenario.eq("baseline"), "duration"].tolist(), [3, 3])
        clear_file = self.root / "products/tables/site-1/summary.csv.gz"
        clear = pd.read_csv(clear_file)
        self.assertEqual(len(clear), 27)
        self.assertEqual(set(clear.period), {"year_2024", "year_2025", "extended_2024_2025"})
        for path in (self.root / "downloads").rglob("*.nc"):
            path.unlink()
        with Registry(self.root) as registry:
            registry.reconcile()
            self.assertEqual(registry.progress().analysis_status.tolist(), ["complete", "complete"])
            self.assertEqual(registry.progress().raw_status.tolist(), ["removed", "removed"])
            with patch.object(full, "download_grib_years") as download:
                self.assertEqual(run(registry, hours=.02, dashboard_address="127.0.0.1:0"), 0)
                download.assert_not_called()
        self.database.unlink()
        pd.testing.assert_frame_equal(history, read_results(self.root, "extended_2024_2025"))
        with Registry(self.root, create=True) as rebuilt:
            rebuilt.reconcile()
            self.assertEqual(rebuilt.progress().analysis_status.tolist(), ["complete", "complete"])
            pd.testing.assert_frame_equal(clear, pd.read_csv(clear_file))
            with patch.object(Registry, "method_signature", return_value="changed-method"):
                rebuilt.reconcile()
                self.assertEqual(rebuilt.progress().analysis_status.tolist(), ["needs_raw", "needs_raw"])
                self.assertEqual(rebuilt.progress().extraction_status.tolist(), ["complete", "complete"])
                self.assertFalse(read_results(self.root, "year_2025").empty)
            rebuilt.reconcile()
        archive = product_path(self.root, 1)
        with gzip.open(archive, "rt") as handle:
            payload = json.load(handle)
        payload["periods"]["year_2025"]["tables"]["summary"] = "damaged"
        with gzip.open(archive, "wt") as handle:
            json.dump(payload, handle)
        with Registry(self.root) as registry:
            registry.reconcile()
            status = registry.db.execute(
                "SELECT status FROM analyses WHERE site_id=1 AND period='year_2025'").fetchone()[0]
            self.assertEqual(status, "failed")
        with self.assertRaises(ValueError):
            read_results(self.root, "year_2025", "unknown")

    def test_single_coordinator_lock_and_zero_budget_are_safe(self):
        self.initialise()
        with Registry(self.root) as first, Registry(self.root) as second:
            with first.lock():
                with self.assertRaisesRegex(RuntimeError, "Another coordinator"):
                    with second.lock():
                        self.fail("Concurrent coordinator acquired lock")
            with patch.object(full, "download_grib_years") as download:
                self.assertEqual(run(first, hours=0), 0)
                download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
