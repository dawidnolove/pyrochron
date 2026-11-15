"""NASA FIRMS NRT ROI queries and shared historical cluster-collapse processing."""
from __future__ import annotations

import io
import math
from datetime import date, datetime, timedelta, timezone
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import numpy as np
import pandas as pd

from .analytics import apply_activity_measure
from .pipeline import (CELL_DEGREES, REGIONS, SENSOR_START, _cluster_aggregate,
                       _monthly_aggregate, _prepare_frame)

DAY_RANGE = 7
SOURCES = {
    'MODIS_NRT': 'MODIS', 'VIIRS_SNPP_NRT': 'SNPP_VIIRS',
    'VIIRS_NOAA20_NRT': 'NOAA20_VIIRS', 'VIIRS_NOAA21_NRT': 'NOAA21_VIIRS',
}
AREA_URL = 'https://firms.modaps.eosdis.nasa.gov/api/area/csv'


def validate_bbox(values):
    try:
        box = tuple(float(value) for value in values)
    except (TypeError, ValueError):
        raise ValueError('bbox must be four numbers: west,south,east,north') from None
    if len(box) != 4 or not all(math.isfinite(x) for x in box):
        raise ValueError('bbox must be four finite numbers: west,south,east,north')
    west, south, east, north = box
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise ValueError('bbox must be ordered west,south,east,north within longitude/latitude limits')
    return box


def _read_csv_window(key, source, bbox, days, date_arg=None, opener=urlopen):
    area = ','.join(f'{v:g}' for v in bbox)
    path = f'{AREA_URL}/{key}/{source}/{area}/{days}'
    if date_arg:
        path += f'/{date_arg}'
    try:
        with opener(path, timeout=25) as response:
            payload = response.read()
    except HTTPError as exc:
        raise RuntimeError(f'{source}: NASA returned HTTP {exc.code}') from None
    except (URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f'{source}: NASA request failed ({type(exc).__name__})') from None
    try:
        frame = pd.read_csv(io.BytesIO(payload))
    except Exception:
        raise RuntimeError(f'{source}: NASA returned an unreadable response') from None
    required = {'latitude', 'longitude', 'acq_date', 'confidence'}
    if not required.issubset(frame.columns):
        raise RuntimeError(f'{source}: NASA response did not contain FIRMS detection columns')
    return frame


def fetch_source_7day(key, source, bbox, today=None, opener=urlopen):
    """Fetch exactly seven UTC calendar days using the Area API's documented 5-day max."""
    today = today or datetime.now(timezone.utc).date()
    recent = _read_csv_window(key, source, bbox, 5, opener=opener)
    older = _read_csv_window(key, source, bbox, 2,
                             date_arg=(today - timedelta(days=6)).isoformat(), opener=opener)
    return pd.concat([recent, older], ignore_index=True).drop_duplicates()


def _pooled_cluster_factors(report):
    factors = {}
    for sensor in ('SNPP_VIIRS', 'NOAA20_VIIRS', 'NOAA21_VIIRS'):
        pairs = [(float(spec[sensor]['cluster_ratio']), float(spec[sensor].get('training_rows', 1)))
                  for spec in report.get('calibration_factors', {}).values()
                  if sensor in spec and np.isfinite(spec[sensor].get('cluster_ratio', np.nan))]
        factors[sensor] = (float(np.average([p[0] for p in pairs],weights=[p[1] for p in pairs]))
                           if pairs else np.nan)
    return factors


def process_roi_sources(source_frames, report, bbox, generated_at=None):
    """Apply pipeline confidence/type filtering, daily cluster collapse, and common HFAI math."""
    generated_at = generated_at or datetime.now(timezone.utc)
    all_rows, daily_frames, source_status = [], [], {}
    available_sensors = set()
    for source, raw in source_frames.items():
        sensor = SOURCES[source]
        frame = _prepare_frame(raw, sensor)
        west, south, east, north = bbox
        frame = frame.loc[frame.longitude.between(west, east) & frame.latitude.between(south, north)].copy()
        source_status[source] = {'status': 'ok', 'quality_filtered_detections': int(len(frame))}
        available_sensors.add(sensor)
        if frame.empty:
            continue
        frame['region'] = 'Custom ROI'
        monthly = _monthly_aggregate(frame)
        clusters = _cluster_aggregate(frame)
        joined = monthly.merge(clusters[['cell_lat_i','cell_lon_i','year','month','cluster_count']],
                               on=['cell_lat_i','cell_lon_i','year','month'], how='left')
        joined['cluster_count'] = joined.cluster_count.fillna(0)
        joined = joined.rename(columns={'count': f'{sensor}_count', 'cluster_count': f'{sensor}_cluster_count'})
        all_rows.append(joined)
        # The same latitude-aware 1 km/date/day-night collapse used by the archive pipeline,
        # retained at acquisition-day resolution for the seven-day operational view.
        if sensor != 'MODIS':
            yy = np.floor(frame.latitude * 110.574).astype('int64')
            xx = np.floor(frame.longitude * 111.32 * np.cos(np.deg2rad(frame.latitude))).astype('int64')
            cells = frame.assign(_ky=yy, _kx=xx).drop_duplicates(['acq_date','daynight','_ky','_kx'])
        else:
            cells = frame
        day = frame.groupby(['cell_lat_i','cell_lon_i','acq_date']).size().rename(f'{sensor}_count').reset_index()
        if sensor != 'MODIS':
            collapsed = cells.groupby(['cell_lat_i','cell_lon_i','acq_date']).size().rename(f'{sensor}_cluster_count').reset_index()
            day = day.merge(collapsed, on=['cell_lat_i','cell_lon_i','acq_date'], how='left')
        daily_frames.append(day)
    if not all_rows:
        return {'source_status': source_status, 'data': [], 'daily': [], 'totals': {}}
    grid = pd.concat(all_rows, ignore_index=True).groupby(
        ['region','cell_lat_i','cell_lon_i','year','month'], as_index=False).sum(numeric_only=True)
    if grid.empty:
        return {'source_status': source_status, 'data': [], 'daily': [], 'totals': {
            'MODIS_count': 0, 'SNPP_VIIRS_count': 0, 'NOAA20_VIIRS_count': 0,
            'NOAA21_VIIRS_count': 0, 'daily_cluster_proxy_sum': 0.0}}
    for sensor in SOURCES.values():
        for suffix in ('count', 'cluster_count'):
            col = f'{sensor}_{suffix}'
            if col not in grid:
                grid[col] = 0
    grid['MODIS_count'] = grid['MODIS_count'].fillna(0)
    # apply_activity_measure keeps the legacy count-aligned comparison alongside the
    # selected cluster measure and expects an input HFAI column for that comparison.
    # Live ROI has no count-aligned baseline, so mark that comparison unavailable.
    grid['hfai'] = np.nan
    factors = _pooled_cluster_factors(report)
    for sensor in ('SNPP_VIIRS','NOAA20_VIIRS','NOAA21_VIIRS'):
        if sensor not in available_sensors:
            factors[sensor] = np.nan
    custom_report = {'calibration_factors': {'Custom ROI': {
        sensor: {'cluster_ratio': value} for sensor, value in factors.items()}}}
    measured = apply_activity_measure(grid, custom_report, 'cluster_collapse',
                                      {'Custom ROI': available_sensors})
    measured['latitude'] = (measured.cell_lat_i + .5) * CELL_DEGREES
    measured['longitude'] = (measured.cell_lon_i + .5) * CELL_DEGREES
    data = measured[['latitude','longitude','hfai','MODIS_count','SNPP_VIIRS_cluster_count',
                     'NOAA20_VIIRS_cluster_count','NOAA21_VIIRS_cluster_count']].to_dict('records')
    totals = {col: int(pd.to_numeric(measured[col], errors='coerce').fillna(0).sum())
              for col in ('MODIS_count','SNPP_VIIRS_count','NOAA20_VIIRS_count','NOAA21_VIIRS_count')}
    totals['daily_cluster_proxy_sum'] = float(measured.hfai.sum())
    daily_grid = pd.concat(daily_frames, ignore_index=True).groupby(
        ['cell_lat_i','cell_lon_i','acq_date'], as_index=False).sum(numeric_only=True)
    for sensor in ('MODIS','SNPP_VIIRS','NOAA20_VIIRS','NOAA21_VIIRS'):
        for suffix in ('count','cluster_count'):
            col=f'{sensor}_{suffix}'
            if col not in daily_grid:
                daily_grid[col]=0
    daily_grid['year']=daily_grid.acq_date.dt.year
    daily_grid['month']=daily_grid.acq_date.dt.month
    daily_grid['hfai']=daily_grid['MODIS_count'].astype(float)
    estimate_parts=[]
    for sensor, factor in factors.items():
        if np.isfinite(factor):
            estimate_parts.append(daily_grid[f'{sensor}_cluster_count'].fillna(0)*factor)
    pieces=([daily_grid.MODIS_count.astype(float)] if 'MODIS' in available_sensors else []) + estimate_parts
    # MODIS and VIIRS estimates are active only for their mission dates.
    estimates=pd.concat(pieces,axis=1)
    estimate_columns=(['MODIS'] if 'MODIS' in available_sensors else [])+[f'VIIRS_{i}' for i in range(len(estimate_parts))]
    estimates.columns=estimate_columns
    active=pd.DataFrame(index=daily_grid.index)
    if 'MODIS' in available_sensors:
        active['MODIS']=daily_grid.acq_date.ge(pd.Timestamp(SENSOR_START['MODIS']))
    active_col=0
    for sensor, factor in factors.items():
        if np.isfinite(factor):
            active[f'VIIRS_{active_col}']=daily_grid.acq_date.ge(pd.Timestamp(SENSOR_START[sensor]))
            active_col+=1
    daily_grid['hfai']=estimates.where(active).mean(axis=1,skipna=True)
    daily = (daily_grid.groupby('acq_date', as_index=False).agg(activity=('hfai','sum'))
             .rename(columns={'acq_date':'date'}).to_dict('records'))
    public_factors={sensor:(float(value) if np.isfinite(value) else None)
                    for sensor,value in factors.items()}
    return {'source_status': source_status, 'data': data, 'daily': daily, 'totals': totals,
            'calibration_factors': public_factors}


def query_roi(key, bbox, report, now=None, fetcher=fetch_source_7day):
    bbox = validate_bbox(bbox)
    now = now or datetime.now(timezone.utc)
    fetched, statuses = {}, {}
    for source in SOURCES:
        try:
            fetched[source] = fetcher(key, source, bbox, today=now.date())
            statuses[source] = {'status': 'ok'}
        except RuntimeError as exc:
            statuses[source] = {'status': 'unavailable', 'message': str(exc)}
    if not fetched:
        raise RuntimeError('All NASA FIRMS NRT sources are unavailable: ' + '; '.join(
            f'{source}: {state["message"]}' for source, state in statuses.items()))
    result = process_roi_sources(fetched, report, bbox, now)
    processed_status=result.get('source_status',{})
    for source,state in statuses.items():
        if state.get('status')=='ok':
            state['quality_filtered_detections']=processed_status.get(source,{}).get('quality_filtered_detections',0)
    result['source_status'] = statuses
    start = now.date() - timedelta(days=6)
    result['meta'] = {
        'mode': 'custom_roi_nrt', 'bbox': bbox, 'generated_at_utc': now.isoformat(),
        'window_start_utc': start.isoformat(), 'window_end_utc': now.date().isoformat(),
        'requested_days': DAY_RANGE, 'area_api_ranges': [5,2], 'measure': 'cluster_collapse',
        'sources': list(SOURCES),
        'calibration_scope': 'Factors are training-row-weighted combinations of the four 2012–2019 benchmark-region fits; this transfer beyond those AOIs is unvalidated.',
        'limitations': 'NRT detections are provisional satellite observations; clusters are an approximate activity proxy, not unique fires or dispatch alerts. NASA Area API accepts bounding boxes; a drawn polygon is queried using its enclosing bbox.'}
    return result
