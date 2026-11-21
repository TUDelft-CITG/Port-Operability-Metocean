# Port operability and metocean screening

## 1. What does this repository contain?

This repository reconstructs the port-operability screening method described in:

- **Wiegel (2020)**, [Mapping port operability indicators across the world](literature/MSc_Thesis_Matijs_Wiegel_Mapping_port_operability_indicators_across_the_world_final.pdf), MSc thesis.
- **Wiegel et al. (2021)**, [Global Mapping of Seaport Operability Risk Indicators Using Open-Source Metocean Data](https://doi.org/10.3390/jmse9070695).

The method estimates how often wind and waves could interrupt operations at
coastal locations. It supports a first comparison of locations, not a prediction
of actual productivity inside a sheltered port.

| File or folder | Purpose |
|---|---|
| [process_global.py](process_global.py) | Collect data and calculate operability in restartable sessions. |
| [analyse-global.ipynb](analyse-global.ipynb) | Inspect progress, causes of inoperability and annual/seasonal results. |
| [analyse-infrastructure.ipynb](analyse-infrastructure.ipynb) | Compare coastal risk with detected port candidates and jetty-size proxies. |
| [operability/](operability/), [port_operability.py](port_operability.py) and [full_study.py](full_study.py) | Shared processing, scientific calculations and plotting code. |
| [literature/](literature/) and [archive/](archive/) | Publications, earlier notebooks and figure-recovery inputs. |

**This is a reconstruction, not an exact numerical reproduction.** The original
manually selected grid and some original inputs are unavailable. The current
grid has 4,013 locations, compared with 4,560 in the study. The publication's
benchmark covers **1979-2018**; the current workflow extends collection to **2025**.

## 2. How does it work?

### Start with coastal ERA5 points

The base grid samples the coast at approximately **55.6 km along-coast spacing**.
Each coastal anchor is matched to a nearby eligible offshore point on ERA5's
native **0.5-degree wave grid**.

The coastal sampling is regular, but the final offshore points are not perfectly
equally spaced: native-grid snapping, islands and straits affect their positions.
They represent coastal exposure, not necessarily existing ports. Greenland and
Antarctica are excluded.

### Collect hourly data

The downloader retrieves [ERA5 hourly single-level data](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download)
from the Copernicus Climate Data Store (CDS).

| ERA5 parameter | What we use it for |
|---|---|
| East-west wind component at 10 m (`u10`) | Calculate wind speed. |
| North-south wind component at 10 m (`v10`) | Calculate wind speed. |
| Significant combined wind-wave and swell height (`swh`) | Offshore wave conditions. |
| Mean wave direction (`mwd`) | Determine how waves approach the coast. |
| Mean wave period (`mwp`) | Estimate wave transformation towards the coast. |
| Peak wave period (`pp1d`) | Estimate infragravity-wave conditions. |

Coastline and bathymetry information provide the coastal orientation and water
depth. These are supporting inputs, not additional hourly weather parameters.

Collection works through all regional groups for the newest year first, then
moves backwards: 2025, 2024, 2023, and so on. Saved requests and completed work
are reused when a session restarts.

### Process the data

1. Split each yearly regional download into **12 monthly NetCDF files**, retaining
   only the coastal points rather than the entire regional grid.
2. Calculate wind speed, transform short waves towards a representative
   **20 m water depth**, and estimate infragravity waves.
3. Calculate availability and stoppage statistics for wind, short waves and
   infragravity waves separately, and for their **overall combined effect**.
4. Save overall, annual and seasonal results, including nine baseline/sensitivity
   scenarios with different thresholds, depths and wind/wave conditions.

**Short + IG waves** is also reported as a *main-driver category* when both wave
effects are similarly restrictive. It is not a fourth independent weather
parameter or a combined availability obtained by adding percentages.

Downloads are sequential; calculations can use several Dask workers. Workers
may therefore be idle while CDS prepares a request.

### Visualise the results

[analyse-global.ipynb](analyse-global.ipynb) shows:

- Download and calculation progress.
- Overall availability and operability risk.
- Main causes, using the colours from paper Figure 6: dark green for wind,
  orange for short waves, magenta for IG waves and cyan for Short + IG waves.
- Four seasonal risk maps and differences from whole-period availability.
- Seasonal availability, event frequency and event duration.

The notebooks read existing results; they do not start downloads. The global
notebook initially selects **2025**, not the publication's 40-year benchmark.
Change `PERIOD`, `SCENARIO` and `COUNTRIES` to explore other available selections.

The optional infrastructure notebook adds candidate sites with clickable
confidence levels. Its jetty lengths are bounding-box size proxies, not surveyed
lengths or proof that a candidate is an actual port.

## 3. How is operability determined?

Think of the method as checking the weather **hour by hour**:

- Is the wind too strong?
- Are the ordinary short waves too high?
- Are the slower infragravity waves, which can move moored ships, too high?

The baseline limits are:

| Factor | Limit |
|---|---|
| Wind speed | 13.8 m/s |
| Short-wave height | 2.0 m |
| Infragravity-wave height | 0.05 m (5 cm) |

An hour is unavailable when **any limit is reached or exceeded**. A calm break
of four hours or less between bad-weather periods is treated as part of the
same stoppage: restarting operations for such a short interval may be impractical.

**Conditional availability** is the percentage of valid observed hours remaining
available after applying this restart rule. The results also describe how often
stoppages occur and their average duration.

| Overall risk | Conditional availability |
|---|---|
| Low | Above 95% |
| Medium | Above 85%, up to 95% |
| High | Above 75%, up to 85% |
| Very high | 75% or less |

The **main driver** is the most restrictive individual factor. If the short-wave
and IG-wave availabilities differ by less than one percentage point, they are
labelled **Short + IG waves**, unless wind is more restrictive. Different causes
can occur at the same time, so their unavailable percentages must not be added.

Missing observations are not assumed to be calm weather. Coverage is reported;
the newer cause/season comparisons default to fully observed sites shared across
all four seasons. DJF, MAM, JJA and SON are calendar seasons: DJF is not winter
in the Southern Hemisphere.

Long-term results use continuous hourly histories, not averages of annual
percentages. Splitting a storm at a year or season boundary can change its event
statistics; seasonal frequencies and durations are not simply additive.

## 4. What do I need to do to run the analysis?

### Install the software

Use Python (3.12 or 3.13 recommended) and a notebook editor, for example VS Code
with its Python and Jupyter extensions. From the repository root, create an
environment and install [requirements-reproduction.txt](requirements-reproduction.txt):

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-reproduction.txt
.venv/bin/python -m eccodes selfcheck
```

These commands use macOS/Linux paths. The requirements include the CDS client,
scientific/geospatial libraries, the ecCodes GRIB decoder, Dask and Folium.
Select **`.venv/bin/python` as the notebook kernel**.

### Create a CDS account and configure credentials

Credentials are needed for new downloads, not for viewing existing results.

1. Register or log in at the [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/).
2. Open the [ERA5 single-level dataset](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download)
   and accept its terms of use.
3. Visit the [CDS API setup page](https://cds.climate.copernicus.eu/how-to-api)
   while logged in and copy your personal access token.
4. Put the following in **`~/.cdsapirc`**, a file in your home directory:

```yaml
url: https://cds.climate.copernicus.eu/api
key: <PERSONAL-ACCESS-TOKEN>
```

Replace the placeholder with your own token. **Keep this file outside the
repository. Never commit credentials or put them in a notebook.**

### Prepare or restore the workspace

If `operability-workspace/` already exists, keep using it; **do not initialise it
again**. Data folders are ignored by Git, so a fresh clone does not include the
downloaded data or an existing workspace.

For a new workspace, first provide these prepared inputs:

- `data/processed/coastal-grid.geojson`: the coastal ERA5 grid.
- `data/reference/ne_110m_admin_0_countries.zip`: the Natural Earth country reference.

The grid-building cell at the end of [archive/output-global.ipynb](archive/output-global.ipynb)
can recreate the grid. Run that cell only; the original model/output cells require
historical external data. The country reference is available from
[Natural Earth](https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip).
Once both inputs are present:

```bash
.venv/bin/python process_global.py init
```

Initialisation freezes the inputs but does not download ERA5. The workspace
holds monthly caches, progress and reusable CSV/gzip results. The live SQLite
database, session logs and temporary transfers are kept outside OneDrive.

If restoring onto another computer, update the `database` path in
`operability-workspace/workspace.json` to a local path, then run:

```bash
.venv/bin/python process_global.py reconcile
```

### Run a bounded collection/processing session

For example, run for up to two hours with four calculation workers:

```bash
.venv/bin/python process_global.py run --hours 2 --workers 4 --max-requests 376
.venv/bin/python process_global.py status
```

`--max-requests` caps new CDS submissions; the time budget will usually stop this
example sooner. No new work is scheduled after the deadline, but an active
transfer or calculation batch can finish later.

Run the same command later to resume. **Use only one processing session per
workspace.** To focus on a small scope, add `--continents Africa --years 2025`.
For an explicitly inspected failure, add `--retry-failed` to retry it.

The script prints a local [Dask dashboard](http://127.0.0.1:8787/status) showing
worker activity. It is available only while the session is running. Full global
collection can take substantial time and disk space; start with a bounded run.

### Open the analysis notebooks

Run [analyse-global.ipynb](analyse-global.ipynb) using the selected environment.
Rerun its results/map cells to refresh them as processing progresses. Choose
`paper_1979_2018` only once that period is listed as available.

Run [analyse-infrastructure.ipynb](analyse-infrastructure.ipynb) for the optional
port-infrastructure comparison; it requires the supplied data under `literature/Vili/`.

**Back up the workspace.** Keep frozen inputs, configuration, method records,
coverage receipts and calculated products. Do not remove hourly caches until all
intended long-term calculations are complete; changed calculations may need
those inputs again. Git is not a substitute for a scientific-data backup.
