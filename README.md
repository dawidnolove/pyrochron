# Pyrochron

**A harmonized fire activity calendar for four study regions.** Pyrochron brings NASA FIRMS MODIS and VIIRS active-fire detections onto a common 0.25° monthly grid, then lets users compare historical activity, inspect sensor records, and export the selected data.

Pyrochron responds to the 2026 NASA Space Apps Challenge **[Harmonization of MODIS and VIIRS Hot Spots](https://www.spaceappschallenge.org/2026/challenges/harmonization-of-modis-and-viirs-hot-spots/)**. It is a working web application and analysis for four bounded case areas: Pantanal, Quebec, Valparaíso, and Punjab.

## Try it

Run both the dashboard and API on Linux, macOS, or Windows with Docker Compose:

```sh
docker compose up --build
```

Open `http://localhost:8501`; the API is available at `http://localhost:8765`. Stop both services with `Ctrl+C` (or run `docker compose down` in another terminal).

For a local Python run without Docker:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
.\launch.ps1
```

The launcher starts the dashboard and local API together. Open the collapsed sidebar from the upper-left control, adjust the filters, then choose **Apply filters**. Date inputs select whole calendar months because the source index is monthly. The dashboard reads the precomputed files in `outputs/data` and can run without the large raw archives. Deploy both the Streamlit frontend and API service together (for example with Docker Compose); the live ROI context depends on the API service. The included Quebec sample point export remains available; custom point exports require the optional FIRMS archive CSVs. The image gallery is in `media/`.

The **Analysis context** switch separates two products. **Historical benchmark regions** uses local precomputed monthly observations for the four configured regions (the available archive period is shown in the dashboard). **Live custom ROI** lets a user draw a rectangle/polygon or enter a bbox, then requests the latest seven UTC days from NASA FIRMS NRT for MODIS, VIIRS S-NPP, NOAA-20, and NOAA-21. NASA's Area API accepts bbox only, so polygons use their enclosing bbox. Live observations pass through the same confidence/type screens and approximate daily 1 km/day-night VIIRS repeat-collapse logic as the historical pipeline. Its comparable-activity calibration transfers benchmark-region fits and is explicitly unvalidated outside those regions; neither the live proxy nor the map is an incident list or dispatch alert.

To enable live FIRMS requests, create a free key through the [NASA FIRMS MAP_KEY page](https://firms.modaps.eosdis.nasa.gov/api/map_key/) and set `FIRMS_MAP_KEY` in `.env`. For CARTO's Positron basemap, set `CARTO_API_KEY` there too; request a basemap key at [CARTO](https://carto.com/basemaps/apikey/). CARTO requires the key as a query parameter on each tile URL, so browsers will see it; restrict the key to your demo site's referrer in CARTO's key dashboard. Docker Compose passes it to the frontend for map rendering. If it is absent, the map falls back to OpenStreetMap. Without a valid FIRMS key, historical mode continues to work and the live API reports that its key is unavailable.

If you start Streamlit directly instead of using the launcher, start the API in a second terminal:

```powershell
python api.py
```

The API defaults to `127.0.0.1:8765`. Set `PYROCHRON_HOST=0.0.0.0` and `PORT` when hosting it. Example: `/api/calendar?bbox=-78,46,-64,55&from=2012-01-01&to=2024-12-31&region=Quebec`; this defaults to `measure=cluster_collapse`. Use `measure=count_aligned_hfai` for count-ratio HFAI. Add `&format=csv` for CSV. The response includes both component series and puts the requested one in `hfai`; its same-month reference is calculated for the requested measure over 2012–2019. `/healthz` reports readiness. Historical requests are limited to the four precomputed AOIs. The separate live route is `/api/roi?bbox=west,south,east,north`; its duration is fixed by the server to seven UTC days and cannot be extended by URL parameters.

NASA's public Area API documents `DAY_RANGE` from 1 to 5. The seven-day request therefore combines a current `/5` query with a dated `/2` query for the preceding days, per sensor. If one source is unavailable, the response reports it and computes from available sensors; if all fail, it returns an upstream error. NRT records are provisional and may be revised by later processing. Keep `.env` private and never expose the key in frontend settings, logs, or a public URL.

## What it measures

The Harmonized Fire Activity Index (HFAI) is a monthly active-fire detection activity measure. The pipeline quality-filters FIRMS detections, grids counts, and uses a count-ratio calibration selected on separate validation years. The selected method is tested on held-out 2020–2024 overlap: 61,441 sensor-region-cell-month comparisons, MAE 5.33 and R² 0.835 against MODIS counts, compared with raw VIIRS MAE 50.83 and R² −14.38. These are count-agreement metrics; they do not establish equal satellite observation opportunity or match individual fires.

The dashboard and API default to **approximate daily 1 km cluster collapse**. This measure collapses repeat VIIRS detections in the same approximate 1 km location, acquisition date, and day/night period before applying its overlap calibration. It scored better than count-ratio on the recorded 2020–2024 holdout (MAE 4.89 and R² 0.920 vs. MAE 5.33 and R² 0.835), so that result now informs the deployment default. The holdout was not used to fit factors; because its ranking informed this default, these scores are descriptive for that product choice and are not an independent post-choice test. The pipeline's separately selected count-aligned HFAI remains available as a comparison; count-ratio had slightly lower MAE in the 2018–2019 method-selection period (3.96 vs. 4.10). Cluster collapse is a repeat-detection proxy; it cannot link an incident across dates or identify unique fires. The sensitivity panel compares each measure against its own 2012–2019 same-month reference. Choose a measure in the sidebar and click **Apply filters**. The weather layer is contextual NASA POWER/MERRA-2 data and does not drive either measure.

**Interpretation limits:** HFAI is not a unique-fire count, burned area, cloud-corrected incidence, fire-cause classification, or forecast probability. Coverage is limited to four request rectangles and the files available in the build. Empty cells mean no qualifying detections in the supplied records, not confirmed absence of fire. Latest archive and provisional NRT dates are shown in the app. See [`outputs/reports/measurement-scope.md`](outputs/reports/measurement-scope.md) and [`outputs/reports/method-qa.md`](outputs/reports/method-qa.md).

## Project structure

- `app.py`, `api.py` — Streamlit dashboard and optional JSON/CSV API.
- `src/pyrochron/` — processing and analytics modules.
- `outputs/data/` — precomputed runtime datasets and validation results.
- `outputs/figures/` — case studies and calibration figures.
- `outputs/reports/` — methodology, data-integrity, and challenge traceability.
- `config/regions.json` — AOI boundaries and FIRMS request IDs.
- `media/` — the four landscape photographs shown in the flip-card gallery.
- `process.txt` — technical keywords, processing workflow, and design/scientific concerns.
- `archives/`, `inputs/` — optional local source data for full rebuilds. Large FIRMS CSVs are intentionally excluded from version control; the precomputed app outputs are included.

The precomputed dashboard is deployable without the source archives. Rebuilding the full pipeline requires the matching FIRMS archive/NRT CSVs under `archives/` and cached NASA POWER data under `inputs/climate/`; see [`bibliography.txt`](bibliography.txt) and [`outputs/reports/data-acquisition-step-by-step.md`](outputs/reports/data-acquisition-step-by-step.md). The original intake remains in this local workspace, while `.gitignore` keeps its large source files out of the public app repository. Run `python -m src.pyrochron.pipeline` to regenerate the derived datasets when the source files are present. The “Recent activity” profile is historical archive context, not a live incident or dispatch feed.

`inputs/land_cover/` is intentionally data-free: the app does not use a land-cover layer, and land cover cannot establish the cause of a hotspot. Its README documents the boundary. The current user filters cover date presets, seasonal month windows, MODIS/VIIRS focus, and day/night source details; see [`process.txt`](process.txt) for the full workflow and known limits.

## Data and references

Data include NASA FIRMS MODIS and VIIRS active-fire products, plus NASA POWER monthly climate context. The exact source products, endpoints, request IDs, acquisition notes, and additional references are recorded in [`bibliography.txt`](bibliography.txt) and [`outputs/reports/source-log.md`](outputs/reports/source-log.md). Source data remain subject to their respective source terms; the project license applies to original project code and documentation.

## Submission checklist

The project files support the submission, but the team must complete the official website form: provide the project summary and challenge selection, a **public working demonstration link**, NASA/partner data used and references, and the required team and originality details. Make sure every team member is registered and listed on the team page. Publish a public source repository and demo before submitting; check that both open without sign-in. The event page lists the deadline as **November 15, 2026, 11:59 p.m. local time**. See the [official 2026 event page](https://www.spaceappschallenge.org/2026/) and the [challenge statement](https://www.spaceappschallenge.org/2026/challenges/harmonization-of-modis-and-viirs-hot-spots/).

## License

Original Pyrochron code and documentation are released under Apache License 2.0; see [`LICENSE`](LICENSE). NASA data and third-party materials retain their own terms and attribution. All sources used in the project are cited in `bibliography.txt`.
