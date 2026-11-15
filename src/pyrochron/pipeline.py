from __future__ import annotations

import json
import math
import re
import sys
from calendar import monthrange
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_DIR = ROOT / 'archives'
POWER_DIR = ROOT / 'inputs' / 'climate'
OUTPUT_DIR = ROOT / 'outputs' / 'data'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REGIONS_CONFIG_PATH = ROOT / 'config' / 'regions.json'
REGIONS: dict[str, dict[str, Any]] = json.loads(REGIONS_CONFIG_PATH.read_text(encoding='utf-8'))
for _name, _spec in REGIONS.items():
    _bbox = _spec.get('bbox', [])
    _ids = _spec.get('request_ids', [])
    if (not _name or len(_bbox) != 4 or not (-180 <= _bbox[0] < _bbox[2] <= 180) or
            not (-90 <= _bbox[1] < _bbox[3] <= 90) or not _ids or
            any(not str(request_id).isdigit() for request_id in _ids)):
        raise ValueError(f'Invalid AOI configuration for {_name!r} in {REGIONS_CONFIG_PATH}')
REQUEST_REGION = {str(rid): region for region, spec in REGIONS.items() for rid in spec['request_ids']}
if len(REQUEST_REGION) != sum(len(spec['request_ids']) for spec in REGIONS.values()):
    raise ValueError(f'Duplicate FIRMS request IDs in {REGIONS_CONFIG_PATH}')
CODE_SENSOR = {'M-C61':'MODIS','SV-C2':'SNPP_VIIRS','J1V-C2':'NOAA20_VIIRS','J2V-C2':'NOAA21_VIIRS'}
SENSOR_LABEL = {'MODIS':'MODIS C6.1','SNPP_VIIRS':'VIIRS S-NPP C2','NOAA20_VIIRS':'VIIRS NOAA-20 C2','NOAA21_VIIRS':'VIIRS NOAA-21 C2'}
SENSOR_START = {'MODIS':'2000-11-01','SNPP_VIIRS':'2012-01-20','NOAA20_VIIRS':'2018-04-01','NOAA21_VIIRS':'2023-03-01'}
SENSORS = list(SENSOR_LABEL)
TRAIN_YEARS = (2012,2019)
TEST_YEARS = (2020,2024)
METHOD_FIT_YEARS = (2012,2017)
METHOD_SELECTION_YEARS = (2018,2019)
MODIS_CONFIDENCE_MIN = 30
VIIRS_CONFIDENCE_KEEP = {'n','h','nominal','high'}
CELL_DEGREES = 0.25
CHUNK_ROWS = 250_000

USECOLS = ['latitude','longitude','brightness','scan','track','acq_date','acq_time','satellite','instrument','confidence','version','bright_t31','frp','daynight','type']

def _file_identity(path: Path) -> tuple[str,str,str] | None:
    m = re.match(r'^fire_(archive|nrt)_(.+)_(\d+)\.csv$', path.name)
    if not m or m.group(3) not in REQUEST_REGION:
        return None
    collection, code, rid = m.groups()
    return collection, code, rid

def _confidence_mask(frame: pd.DataFrame, sensor: str) -> pd.Series:
    value = frame.get('confidence', pd.Series('', index=frame.index)).astype(str).str.strip().str.lower()
    if sensor == 'MODIS':
        numeric = pd.to_numeric(value, errors='coerce')
        return numeric.ge(MODIS_CONFIDENCE_MIN)
    return value.isin(VIIRS_CONFIDENCE_KEEP)

def _prepare_frame(frame: pd.DataFrame, sensor: str) -> pd.DataFrame:
    frame = frame.copy()
    for col in ['latitude','longitude','frp','scan','track','brightness','bright_t31']:
        if col not in frame:
            frame[col] = np.nan
        frame[col] = pd.to_numeric(frame[col], errors='coerce')
    frame['acq_date'] = pd.to_datetime(frame['acq_date'], errors='coerce').dt.normalize()
    frame = frame.loc[frame['acq_date'].notna() & frame.latitude.between(-90,90) & frame.longitude.between(-180,180)].copy()
    frame = frame.loc[_confidence_mask(frame, sensor)].copy()
    if 'type' in frame:
        fire_type = pd.to_numeric(frame['type'], errors='coerce')
        # Apply FIRMS type 0 as a source-field screen; it is not a validated fire-regime label.
        frame = frame.loc[fire_type.eq(0)].copy()
    # NRT archive companions do not always carry a type column; those rows remain marked in source quality metadata.
    frame['daynight'] = frame.get('daynight', pd.Series('',index=frame.index)).astype(str).str.upper().str[:1]
    frame.loc[~frame.daynight.isin(['D','N']), 'daynight'] = 'U'
    frame['cell_lat_i'] = np.floor(frame.latitude / CELL_DEGREES).astype('int32')
    frame['cell_lon_i'] = np.floor(frame.longitude / CELL_DEGREES).astype('int32')
    frame['year'] = frame.acq_date.dt.year.astype('int16')
    frame['month'] = frame.acq_date.dt.month.astype('int8')
    frame['source'] = sensor
    frame['confidence_raw'] = frame.get('confidence', '').astype(str) if isinstance(frame.get('confidence'), pd.Series) else ''
    return frame

def _monthly_aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=['region','sensor','cell_lat_i','cell_lon_i','year','month','count','frp_mw','confidence_filtered_count','night_count','day_count'])
    frame=frame.assign(_night=frame.daynight.eq('N').astype('int8'),_day=frame.daynight.eq('D').astype('int8'))
    g = frame.groupby(['cell_lat_i','cell_lon_i','year','month'], observed=True, sort=False)
    out = g.agg(count=('latitude','size'), frp_mw=('frp','sum'), night_count=('_night','sum'), day_count=('_day','sum')).reset_index()
    if frame['source'].iloc[0]=='MODIS' and 'satellite' in frame:
        labels=frame.satellite.astype(str).str.strip().str.casefold()
        for platform,aliases in {'Terra':{'terra','t'},'Aqua':{'aqua','a'}}.items():
            part=frame.loc[labels.isin(aliases)].copy()
            part[f'_{platform.lower()}_night']=part.daynight.eq('N').astype('int8')
            part[f'_{platform.lower()}_day']=part.daynight.eq('D').astype('int8')
            pg=part.groupby(['cell_lat_i','cell_lon_i','year','month'],observed=True,sort=False).agg(
                **{f'MODIS_{platform}_count':('latitude','size'),f'MODIS_{platform}_day_count':(f'_{platform.lower()}_day','sum'),f'MODIS_{platform}_night_count':(f'_{platform.lower()}_night','sum')}).reset_index()
            out=out.merge(pg,on=['cell_lat_i','cell_lon_i','year','month'],how='left')
    out['region'] = frame['region'].iloc[0]
    out['sensor'] = frame['source'].iloc[0]
    return out

def _cluster_aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or frame.source.iloc[0] == 'MODIS':
        return pd.DataFrame(columns=['region','sensor','cell_lat_i','cell_lon_i','year','month','cluster_count'])
    # Approximate 1-km footprint grid (latitude-aware equirectangular cells); collapse repeat detections within date and day/night.
    yy = np.floor(frame.latitude * 110.574).astype('int64')
    xx = np.floor(frame.longitude * 111.32 * np.cos(np.deg2rad(frame.latitude))).astype('int64')
    clusters = frame.assign(_ky=yy,_kx=xx).drop_duplicates(['acq_date','daynight','_ky','_kx'])
    out = clusters.groupby(['cell_lat_i','cell_lon_i','year','month'], observed=True, sort=False).size().rename('cluster_count').reset_index()
    out['region'] = frame.region.iloc[0]
    out['sensor'] = frame.source.iloc[0]
    return out

def _read_source_file(path: Path, sensor: str) -> pd.DataFrame:
    # Read one source at a time to keep memory bounded; discard non-analysis metadata only after retaining all audited FIRMS fields.
    try:
        cols = pd.read_csv(path, nrows=0).columns.tolist()
    except Exception:
        return pd.DataFrame()
    usecols = [c for c in USECOLS if c in cols]
    raw = pd.read_csv(path, usecols=usecols, low_memory=False)
    return _prepare_frame(raw, sensor)

def _source_has_type(path: Path) -> bool:
    try:
        return 'type' in pd.read_csv(path, nrows=0).columns
    except Exception:
        return False

def _after_archive_cutoff(frame: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    return frame.loc[frame.acq_date.gt(pd.Timestamp(cutoff))].copy()

def _mission_month_mask(frame: pd.DataFrame, sensor: str) -> pd.Series:
    starts=pd.to_datetime(dict(year=frame.year,month=frame.month,day=1))
    mission_month=pd.Timestamp(SENSOR_START[sensor]).to_period('M').start_time
    return starts.ge(mission_month)

def _factor(target: pd.Series, predictor: pd.Series) -> float:
    denominator = float(pd.to_numeric(predictor, errors='coerce').fillna(0).sum())
    numerator = float(pd.to_numeric(target, errors='coerce').fillna(0).sum())
    if denominator <= 0 or numerator <= 0:
        return float('nan')
    return numerator / denominator

def _metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    valid = np.isfinite(observed) & np.isfinite(predicted)
    y = observed[valid].astype(float); p = predicted[valid].astype(float)
    if not len(y):
        return {'n':0,'r2':None,'mae':None,'rmse':None,'bias':None,'bias_pct':None,'sse':None,'sst':None,'sum_y':None,'sum_y2':None}
    err = p-y
    denom = float(np.sum((y-y.mean())**2))
    r2 = float(1 - np.sum(err**2)/denom) if denom > 0 else None
    mean_y = float(y.mean())
    return {'n':int(len(y)),'r2':r2,'mae':float(np.mean(np.abs(err))),
            'rmse':float(np.sqrt(np.mean(err**2))),'bias':float(np.mean(err)),
            'bias_pct':float(np.mean(err)/mean_y*100) if mean_y else None,
            'sse':float(np.sum(err**2)),'sst':denom,'sum_y':float(np.sum(y)),'sum_y2':float(np.sum(y*y))}

def _select_method_before_final_holdout(monthly: pd.DataFrame, available_archive: set[tuple[str,str]]) -> tuple[str,list[dict[str,Any]]]:
    """Fit on 2012-17, select on 2018-19, and never inspect the 2020-24 holdout."""
    methods=[('count_ratio','count','count_ratio'),('cluster_collapse','cluster_count','cluster_ratio'),
             ('frp_proxy','frp_mw','frp_to_modis_count')]
    residual_pairs={method:[] for method,_,_ in methods}
    years=[]
    for region in REGIONS:
        group=monthly.loc[monthly.region.eq(region)]
        fit=group.loc[group.year.between(*METHOD_FIT_YEARS)]
        select=group.loc[group.year.between(*METHOD_SELECTION_YEARS)]
        family={key:0.0 for _method,_field,key in methods}
        for sensor in ('SNPP_VIIRS','NOAA20_VIIRS'):
            if (region,sensor) not in available_archive: continue
            paired=fit.loc[_mission_month_mask(fit,sensor)]
            family['count_ratio']+=float(paired[f'{sensor}_count'].sum())
            family['cluster_ratio']+=float(paired[f'{sensor}_cluster_count'].sum())
            family['frp_to_modis_count']+=float(paired[f'{sensor}_frp_mw'].sum())
        family_target=sum(float(fit.loc[_mission_month_mask(fit,s), 'MODIS_count'].sum())
                          for s in ('SNPP_VIIRS','NOAA20_VIIRS') if (region,s) in available_archive)
        family_factors={key:_factor(pd.Series([family_target]),pd.Series([denom])) for key,denom in family.items()}
        for sensor in ('SNPP_VIIRS','NOAA20_VIIRS','NOAA21_VIIRS'):
            if (region,sensor) not in available_archive: continue
            sensor_fit=fit.loc[_mission_month_mask(fit,sensor)]
            factors={}
            for _method,field,key in methods:
                factors[key]=_factor(sensor_fit['MODIS_count'],sensor_fit[f'{sensor}_{field}'])
                if not np.isfinite(factors[key]): factors[key]=family_factors[key]
            active=select.loc[_mission_month_mask(select,sensor)]
            valid=(active.MODIS_count.gt(0)|active[f'{sensor}_count'].gt(0)|
                   active[f'{sensor}_cluster_count'].gt(0)|active[f'{sensor}_frp_mw'].gt(0))
            active=active.loc[valid]
            if active.empty: continue
            observed=active.MODIS_count.to_numpy(dtype=float)
            for method,field,key in methods:
                factor=factors[key]
                predicted=active[f'{sensor}_{field}'].to_numpy(dtype=float)*(float(factor) if np.isfinite(factor) else 0.0)
                residual_pairs[method].append((observed,predicted))
    scores=[]
    for method,_,_ in methods:
        pairs=residual_pairs[method]
        observed=np.concatenate([x[0] for x in pairs]) if pairs else np.array([],dtype=float)
        predicted=np.concatenate([x[1] for x in pairs]) if pairs else np.array([],dtype=float)
        scores.append({'method':method,**_metrics(observed,predicted),
                       'fit_years':list(METHOD_FIT_YEARS),'selection_years':list(METHOD_SELECTION_YEARS)})
    eligible=[x for x in scores if x['mae'] is not None]
    selected=min(eligible,key=lambda x:(x['mae'],x['method']))['method'] if eligible else 'cluster_collapse'
    return selected,scores

def _power_monthly() -> pd.DataFrame:
    rows=[]
    for path in sorted(POWER_DIR.glob('nasa_power_monthly_*.json')):
        data=json.loads(path.read_text(encoding='utf-8'))
        region=data.get('_pyrochron',{}).get('region')
        if region not in REGIONS: continue
        params=data.get('properties',{}).get('parameter',{})
        temps=params.get('T2M',{}); rain=params.get('PRECTOTCORR',{})
        for key,tval in temps.items():
            # POWER annual summary uses month 13; use calendar-month entries only.
            if not re.fullmatch(r'\d{4}(0[1-9]|1[0-2])',key): continue
            year=int(key[:4]); month=int(key[4:])
            rv=rain.get(key)
            temp=float(tval); rain_daily=float(rv) if rv is not None else np.nan
            if temp <= -900 or rain_daily <= -900: continue
            rows.append({'region':region,'year':year,'month':month,'power_t2m_c':temp,
                         'power_precip_mm_day':rain_daily,'power_precip_mm_month':rain_daily*monthrange(year,month)[1]})
    weather=pd.DataFrame(rows)
    if weather.empty: return weather
    baseline=weather[weather.year.between(2001,2020)].groupby(['region','month']).agg(
        precip_clim_mm=('power_precip_mm_month','median'),temp_clim_c=('power_t2m_c','median'),
        precip_sd=('power_precip_mm_month','std'),temp_sd=('power_t2m_c','std')).reset_index()
    weather=weather.merge(baseline,on=['region','month'],how='left')
    weather['power_precip_anomaly_mm']=weather.power_precip_mm_month-weather.precip_clim_mm
    # Relative percentages explode when the normal monthly rainfall is near zero; keep those cases in absolute mm only.
    weather['power_precip_anomaly_pct']=np.where(weather.precip_clim_mm>=10.0,100*weather.power_precip_anomaly_mm/weather.precip_clim_mm,np.nan)
    weather['power_temp_anomaly_c']=weather.power_t2m_c-weather.temp_clim_c
    weather['power_precip_z']=np.where(weather.precip_sd>0,weather.power_precip_anomaly_mm/weather.precip_sd,np.nan)
    return weather.drop(columns=['precip_clim_mm','temp_clim_c','precip_sd','temp_sd'])

def build() -> dict[str,Any]:
    source_frames=[]; cluster_frames=[]; file_audit=[]
    archive_paths=sorted(ARCHIVE_DIR.glob('fire_archive_*.csv'))
    discovered=[]
    for path in archive_paths:
        ident=_file_identity(path)
        if ident:
            collection,code,rid=ident
            discovered.append((path,collection,code,rid,REQUEST_REGION[rid],CODE_SENSOR.get(code)))
    latest_archive={}
    # Read canonical science/archive records first. NRT tail rows are incorporated only after each sensor's standard archive cutoff.
    for path,collection,code,rid,region,sensor in discovered:
        if collection!='archive' or sensor is None: continue
        frame=_read_source_file(path,sensor)
        if frame.empty:
            latest_archive[(region,sensor)]=None
            continue
        frame['region']=region
        dates=frame.acq_date
        cutoff=dates.max()
        latest_archive[(region,sensor)]=cutoff
        source_frames.append(_monthly_aggregate(frame))
        cluster_frames.append(_cluster_aggregate(frame))
        file_audit.append({'file':path.name,'request_id':rid,'region':region,'sensor':sensor,'rows_after_quality_filter':int(len(frame)),
                           'date_min':str(dates.min().date()),'date_max':str(cutoff.date()),'source_quality':'archive','type_field_present':_source_has_type(path)} )
        del frame
    # Add only the newer portion of NRT companions. NRT rows whose dates overlap the archive cutoff are discarded to avoid duplicates.
    for path in sorted(ARCHIVE_DIR.glob('fire_nrt_*.csv')):
        ident=_file_identity(path)
        if not ident: continue
        collection,code,rid=ident; region=REQUEST_REGION[rid]; sensor=CODE_SENSOR.get(code)
        if not sensor: continue
        cutoff=latest_archive.get((region,sensor))
        if cutoff is None: continue
        frame=_read_source_file(path,sensor)
        if frame.empty: continue
        frame=_after_archive_cutoff(frame,cutoff)
        if frame.empty: continue
        frame['region']=region
        source_frames.append(_monthly_aggregate(frame)); cluster_frames.append(_cluster_aggregate(frame))
        file_audit.append({'file':path.name,'request_id':rid,'region':region,'sensor':sensor,'rows_after_quality_filter':int(len(frame)),
                           'date_min':str(frame.acq_date.min().date()),'date_max':str(frame.acq_date.max().date()),
                           'archive_cutoff':str(cutoff.date()),'source_quality':'NRT after archive cutoff','type_field_present':_source_has_type(path)})
        del frame
    if not source_frames: raise RuntimeError('No FIRMS archive rows were found. Check the archives/ folder.')
    observations=pd.concat(source_frames,ignore_index=True)
    obs_agg={'count':('count','sum'),'frp_mw':('frp_mw','sum'),'night_count':('night_count','sum'),'day_count':('day_count','sum')}
    for col in ('MODIS_Terra_count','MODIS_Aqua_count','MODIS_Terra_day_count','MODIS_Terra_night_count','MODIS_Aqua_day_count','MODIS_Aqua_night_count'):
        if col in observations: obs_agg[col]=(col,'sum')
    observations=(observations.groupby(['region','sensor','cell_lat_i','cell_lon_i','year','month'],as_index=False,observed=True).agg(**obs_agg))
    clusters=pd.concat([x for x in cluster_frames if not x.empty],ignore_index=True) if any(not x.empty for x in cluster_frames) else pd.DataFrame(columns=['region','sensor','cell_lat_i','cell_lon_i','year','month','cluster_count'])
    if not clusters.empty:
        clusters=(clusters.groupby(['region','sensor','cell_lat_i','cell_lon_i','year','month'],as_index=False,observed=True)
                  .agg(cluster_count=('cluster_count','sum')))
    # Create a complete monthly record for every cell ever observed in a case region; no-detection months are zeros only inside sensor coverage.
    merged=observations.merge(clusters,on=['region','sensor','cell_lat_i','cell_lon_i','year','month'],how='left')
    merged['cluster_count']=merged.cluster_count.fillna(0)
    count_wide=merged.pivot_table(index=['region','cell_lat_i','cell_lon_i','year','month'],columns='sensor',values='count',aggfunc='sum',fill_value=0).reset_index()
    frp_wide=merged.pivot_table(index=['region','cell_lat_i','cell_lon_i','year','month'],columns='sensor',values='frp_mw',aggfunc='sum',fill_value=0).reset_index()
    night_wide=merged.pivot_table(index=['region','cell_lat_i','cell_lon_i','year','month'],columns='sensor',values='night_count',aggfunc='sum',fill_value=0).reset_index()
    day_wide=merged.pivot_table(index=['region','cell_lat_i','cell_lon_i','year','month'],columns='sensor',values='day_count',aggfunc='sum',fill_value=0).reset_index()
    cluster_wide=merged.pivot_table(index=['region','cell_lat_i','cell_lon_i','year','month'],columns='sensor',values='cluster_count',aggfunc='sum',fill_value=0).reset_index()
    def rename_wide(df,suffix):
        df.columns=[c if c in ('region','cell_lat_i','cell_lon_i','year','month') else f'{c}_{suffix}' for c in df.columns]
        return df
    monthly=rename_wide(count_wide,'count').merge(rename_wide(frp_wide,'frp_mw'),on=['region','cell_lat_i','cell_lon_i','year','month'],how='outer')
    monthly=monthly.merge(rename_wide(night_wide,'night_count'),on=['region','cell_lat_i','cell_lon_i','year','month'],how='outer')
    monthly=monthly.merge(rename_wide(day_wide,'day_count'),on=['region','cell_lat_i','cell_lon_i','year','month'],how='outer')
    monthly=monthly.merge(rename_wide(cluster_wide,'cluster_count'),on=['region','cell_lat_i','cell_lon_i','year','month'],how='outer')
    platform_cols=[c for c in ('MODIS_Terra_count','MODIS_Aqua_count','MODIS_Terra_day_count','MODIS_Terra_night_count','MODIS_Aqua_day_count','MODIS_Aqua_night_count') if c in merged]
    if platform_cols:
        platform_monthly=merged.groupby(['region','cell_lat_i','cell_lon_i','year','month'],as_index=False,observed=True)[platform_cols].sum()
        monthly=monthly.merge(platform_monthly,on=['region','cell_lat_i','cell_lon_i','year','month'],how='left')
    for sensor in SENSORS:
        for suffix in ('count','frp_mw','cluster_count','night_count','day_count'):
            name=f'{sensor}_{suffix}'
            if name not in monthly: monthly[name]=0.0
            monthly[name]=pd.to_numeric(monthly[name],errors='coerce').fillna(0)
    # The candidate calibration table uses only matched spatial-cell × month observations in fixed training years.
    factors={region:{} for region in REGIONS}
    calibration_rows=[]
    available_archive={(x['region'],x['sensor']) for x in file_audit if x['source_quality']=='archive'}
    # Select the harmonization method on 2018-19 before opening the final 2020-24 holdout.
    selected,method_selection_scores=_select_method_before_final_holdout(monthly,available_archive)
    for region in REGIONS:
        group=monthly[monthly.region.eq(region)].copy()
        train=group[group.year.between(*TRAIN_YEARS)].copy()
        train['period_start']=pd.to_datetime(dict(year=train.year,month=train.month,day=1))
        # Shared VIIRS-C2 family factors are the pre-2020 transfer basis for newer NOAA-21; no NOAA-21 training observations leak into the fit.
        family_count_num=family_count_den=family_cluster_den=family_frp_den=0.0
        for sensor in ('SNPP_VIIRS','NOAA20_VIIRS'):
            sensor_train=train.loc[_mission_month_mask(train,sensor)]
            if (region,sensor) in available_archive:
                family_count_num += float(sensor_train['MODIS_count'].sum())
                family_count_den += float(sensor_train[f'{sensor}_count'].sum())
                family_cluster_den += float(sensor_train[f'{sensor}_cluster_count'].sum())
                family_frp_den += float(sensor_train[f'{sensor}_frp_mw'].sum())
        # For a sensor-specific fit, each cell-month is paired against MODIS on the same 0.25-degree lattice.
        for sensor in ('SNPP_VIIRS','NOAA20_VIIRS','NOAA21_VIIRS'):
            if sensor == 'NOAA21_VIIRS':
                raw_count_factor=_factor(pd.Series([family_count_num]),pd.Series([family_count_den]))
                cluster_factor=_factor(pd.Series([family_count_num]),pd.Series([family_cluster_den]))
                frp_factor=_factor(pd.Series([family_count_num]),pd.Series([family_frp_den]))
                family_rows=train.loc[_mission_month_mask(train,'SNPP_VIIRS') &
                    ((train.SNPP_VIIRS_count.gt(0)|train.SNPP_VIIRS_cluster_count.gt(0)|train.SNPP_VIIRS_frp_mw.gt(0)) |
                     (train.NOAA20_VIIRS_count.gt(0)|train.NOAA20_VIIRS_cluster_count.gt(0)|train.NOAA20_VIIRS_frp_mw.gt(0)) |
                     train.MODIS_count.gt(0))]
                fit_rows=int(len(family_rows))
                fit_basis='pooled S-NPP + NOAA-20 VIIRS C2 training (NOAA-21 transfer; no fit leakage)'
            else:
                sensor_train=train.loc[_mission_month_mask(train,sensor)]
                raw_count_factor=_factor(sensor_train['MODIS_count'],sensor_train[f'{sensor}_count'])
                cluster_factor=_factor(sensor_train['MODIS_count'],sensor_train[f'{sensor}_cluster_count'])
                frp_factor=_factor(sensor_train['MODIS_count'],sensor_train[f'{sensor}_frp_mw'])
                fit_rows=int((sensor_train['MODIS_count'].gt(0)|sensor_train[f'{sensor}_count'].gt(0)).sum())
                fit_basis='sensor-specific matched cell-month training'
            # If a source lacks sufficient count data, transfer a pooled pre-2020 VIIRS-C2 calibration.
            if not np.isfinite(raw_count_factor): raw_count_factor=_factor(pd.Series([family_count_num]),pd.Series([family_count_den]))
            if not np.isfinite(cluster_factor): cluster_factor=_factor(pd.Series([family_count_num]),pd.Series([family_cluster_den]))
            if not np.isfinite(frp_factor): frp_factor=_factor(pd.Series([family_count_num]),pd.Series([family_frp_den]))
            factors[region][sensor]={'count_ratio':raw_count_factor,'cluster_ratio':cluster_factor,'frp_to_modis_count':frp_factor,
                                     'training_rows':fit_rows,'training_years':list(TRAIN_YEARS),'fit_basis':fit_basis}
        # Candidate models are scored against held-out MODIS counts from the same cells and months.
        test=group[group.year.between(*TEST_YEARS)].copy()
        test['period_start']=pd.to_datetime(dict(year=test.year,month=test.month,day=1))
        for method,field,factor_key in [('count_ratio','count','count_ratio'),('cluster_collapse','cluster_count','cluster_ratio'),('frp_proxy','frp_mw','frp_to_modis_count')]:
            for sensor in ('SNPP_VIIRS','NOAA20_VIIRS','NOAA21_VIIRS'):
                factor=factors[region][sensor].get(factor_key)
                active_test=test.loc[_mission_month_mask(test,sensor)] if (region,sensor) in available_archive else test.iloc[0:0]
                valid=active_test['MODIS_count'].gt(0)|active_test[f'{sensor}_count'].gt(0)|active_test[f'{sensor}_cluster_count'].gt(0)|active_test[f'{sensor}_frp_mw'].gt(0)
                subset=active_test.loc[valid]
                pred=subset[f'{sensor}_{field}'].to_numpy(dtype=float)*float(factor or 0.0)
                metric=_metrics(subset['MODIS_count'].to_numpy(dtype=float),pred)
                calibration_rows.append({'region':region,'method':method,'sensor':sensor,'factor':float(factor) if np.isfinite(factor) else None,**metric})
    # Candidate selection is based on true pooled out-of-time errors, not averages
    # of region-level scores (which can distort the result when errors vary).
    validation=[]
    caldf=pd.DataFrame(calibration_rows)
    for method,part in caldf.groupby('method'):
        scored=part[part.n.gt(0)]
        # Recreate pooled sufficient statistics from pairwise metrics and n.
        total_n=int(scored.n.sum())
        pooled_mae=float(np.average(scored.mae,weights=scored.n)) if total_n else None
        pooled_rmse=float(np.sqrt(np.average(scored.rmse.pow(2),weights=scored.n))) if total_n else None
        pooled_bias=float(np.average(scored.bias,weights=scored.n)) if total_n else None
        valid_sse=scored.sse.dropna().sum(); sum_y=scored.sum_y.dropna().sum(); sum_y2=scored.sum_y2.dropna().sum()
        pooled_sst=float(sum_y2-sum_y**2/total_n) if total_n else 0
        validation.append({'method':method,'pooled_holdout_mae':pooled_mae,
                           'pooled_holdout_rmse':pooled_rmse,'pooled_holdout_bias':pooled_bias,
                           'pooled_holdout_r2':float(1-valid_sse/pooled_sst) if pooled_sst>0 else None,
                           'pooled_n':total_n,
                           'macro_holdout_mae':float(scored.mae.mean()) if total_n else None,
                           'macro_holdout_rmse':float(scored.rmse.mean()) if total_n else None,
                           'macro_holdout_r2':float(scored.r2.mean()) if scored.r2.notna().any() else None,
                           'macro_holdout_bias':float(scored.bias.mean()) if total_n else None,
                           'macro_absolute_bias_pct':float(scored.bias_pct.abs().mean()) if scored.bias_pct.notna().any() else None,
                           'scored_sensor_region_pairs':int(len(scored))})
    field_by_method={'count_ratio':'count_ratio','cluster_collapse':'cluster_ratio','frp_proxy':'frp_to_modis_count'}
    estimate_cols=[]
    for sensor in ('SNPP_VIIRS','NOAA20_VIIRS','NOAA21_VIIRS'):
        monthly[f'{sensor}_modis_equivalent']=np.nan
        for region in REGIONS:
            factor=factors[region].get(sensor,{}).get(field_by_method[selected])
            mask=monthly.region.eq(region)
            monthly.loc[mask,f'{sensor}_modis_equivalent']=monthly.loc[mask,f'{sensor}_{selected_field(selected)}']*float(factor) if np.isfinite(factor) else np.nan
        estimate_cols.append(f'{sensor}_modis_equivalent')
    monthly['MODIS_modis_equivalent']=monthly['MODIS_count'].astype(float)
    # Preserve the active-sensor denominator; do not count a source before its mission coverage began.
    for sensor in SENSORS:
        active_start=pd.Timestamp(SENSOR_START[sensor])
        available_pairs={(x['region'],x['sensor']) for x in file_audit if x['source_quality']=='archive'}
        available_regions={r for r,s in available_pairs if s==sensor}
        present=monthly.region.isin(available_regions)
        date=pd.to_datetime(dict(year=monthly.year,month=monthly.month,day=1))
        active=present & date.ge(active_start.replace(day=1))
        monthly[f'{sensor}_active']=active
        monthly[f'{sensor}_estimate_active']=np.where(active,monthly[f'{sensor}_modis_equivalent'],np.nan)
        # When the source is active, a missing point detection is a zero in that cell-month.
        monthly.loc[active & monthly[f'{sensor}_estimate_active'].isna(),f'{sensor}_estimate_active']=0.0
    active_cols=[f'{s}_estimate_active' for s in SENSORS]
    monthly['active_sensor_count']=monthly[active_cols].notna().sum(axis=1).astype('int8')
    monthly['hfai']=monthly[active_cols].mean(axis=1,skipna=True)
    monthly['hfai_method']=selected
    monthly['lat']=monthly.cell_lat_i.astype(float)*CELL_DEGREES+CELL_DEGREES/2
    monthly['lon']=monthly.cell_lon_i.astype(float)*CELL_DEGREES+CELL_DEGREES/2
    monthly['month_start']=pd.to_datetime(dict(year=monthly.year,month=monthly.month,day=1))
    monthly['raw_sensor_detections']=monthly[[f'{s}_count' for s in SENSORS]].sum(axis=1)
    # Expand zero-detection months for every cell observed at least once in a case region.
    dense=[]
    for region,part in monthly.groupby('region',sort=False):
        cells=part[['cell_lat_i','cell_lon_i']].drop_duplicates()
        dates=pd.DataFrame({'month_start':pd.date_range('2000-11-01','2026-11-01',freq='MS')})
        dates['year']=dates.month_start.dt.year.astype('int16'); dates['month']=dates.month_start.dt.month.astype('int8')
        cells['_j']=1; dates['_j']=1
        grid=cells.merge(dates,on='_j').drop(columns='_j')
        grid['region']=region
        grid=grid.merge(part.drop(columns=['lat','lon','month_start']),on=['region','cell_lat_i','cell_lon_i','year','month'],how='left')
        for col in part.columns:
            if col.endswith('_count') or col.endswith('_frp_mw') or col.endswith('_cluster_count') or col in ('raw_sensor_detections','active_sensor_count'):
                if col in grid: grid[col]=grid[col].fillna(0)
        for col in active_cols:
            if col in grid:
                # Calculate active/available sensor status for zero rows as well.
                sensor=col.removesuffix('_estimate_active')
                has_archive=any(r==region and s==sensor for r,s in available_pairs)
                if has_archive:
                    start=pd.Timestamp(SENSOR_START[sensor]).replace(day=1)
                    available=grid.month_start.ge(start)
                    if sensor=='SNPP_VIIRS': available &= grid.month_start.le(pd.Timestamp('2026-11-01'))
                    grid[col]=grid[col].where(~available,grid[col].fillna(0))
                    grid.loc[~available,col]=np.nan
                else: grid[col]=np.nan
        grid['active_sensor_count']=grid[active_cols].notna().sum(axis=1).astype('int8')
        grid['hfai']=grid[active_cols].mean(axis=1,skipna=True)
        grid['hfai_method']=selected
        grid['lat']=grid.cell_lat_i.astype(float)*CELL_DEGREES+CELL_DEGREES/2
        grid['lon']=grid.cell_lon_i.astype(float)*CELL_DEGREES+CELL_DEGREES/2
        grid['raw_sensor_detections']=grid[[f'{s}_count' for s in SENSORS]].sum(axis=1)
        dense.append(grid)
    monthly=pd.concat(dense,ignore_index=True)
    # Point-centroid weather context from NASA POWER MERRA-2: useful association only, not a causal fire-weather model.
    weather=_power_monthly()
    if not weather.empty:
        monthly=monthly.merge(weather,on=['region','year','month'],how='left')
    # Build the regional calendar first; same-month medians are computed across yearly regional totals.
    agg={c:'sum' for c in ['MODIS_count','SNPP_VIIRS_count','NOAA20_VIIRS_count','NOAA21_VIIRS_count','MODIS_Terra_count','MODIS_Aqua_count','MODIS_Terra_day_count','MODIS_Terra_night_count','MODIS_Aqua_day_count','MODIS_Aqua_night_count','raw_sensor_detections','hfai'] if c in monthly}
    agg.update({'active_sensor_count':'max'})
    for c in ['power_t2m_c','power_temp_anomaly_c','power_precip_mm_day','power_precip_mm_month','power_precip_anomaly_mm','power_precip_anomaly_pct','power_precip_z']:
        if c in monthly: agg[c]='mean'
    calendar=monthly.groupby(['region','year','month'],as_index=False).agg(agg)
    base=calendar[calendar.year.between(2012,2019)].groupby(['region','month']).agg(
        hfai_month_median=('hfai','median'),hfai_month_p75=('hfai',lambda s:float(s.quantile(.75))),
        hfai_month_n=('hfai','count'),raw_month_median=('raw_sensor_detections','median')).reset_index()
    calendar=calendar.merge(base,on=['region','month'],how='left')
    calendar['hfai_vs_median_ratio']=np.where(calendar.hfai_month_median>0,calendar.hfai/calendar.hfai_month_median,np.nan)
    calendar['hfai_anomaly_pct']=np.where(calendar.hfai_month_median>0,100*(calendar.hfai-calendar.hfai_month_median)/calendar.hfai_month_median,np.nan)
    monthly=monthly.merge(base,on=['region','month'],how='left')
    # Cell-centre geometry makes a bbox query exact at the declared output resolution.
    front=['region','year','month','month_start','cell_lat_i','cell_lon_i','lat','lon','MODIS_count','SNPP_VIIRS_count','NOAA20_VIIRS_count','NOAA21_VIIRS_count',
           'MODIS_frp_mw','SNPP_VIIRS_frp_mw','NOAA20_VIIRS_frp_mw','NOAA21_VIIRS_frp_mw',
           'SNPP_VIIRS_cluster_count','NOAA20_VIIRS_cluster_count','NOAA21_VIIRS_cluster_count',
           'MODIS_estimate_active','SNPP_VIIRS_estimate_active','NOAA20_VIIRS_estimate_active','NOAA21_VIIRS_estimate_active',
           'MODIS_day_count','MODIS_night_count','MODIS_Terra_count','MODIS_Aqua_count','MODIS_Terra_day_count','MODIS_Terra_night_count','MODIS_Aqua_day_count','MODIS_Aqua_night_count',
           'SNPP_VIIRS_day_count','SNPP_VIIRS_night_count','NOAA20_VIIRS_day_count','NOAA20_VIIRS_night_count','NOAA21_VIIRS_day_count','NOAA21_VIIRS_night_count',
           'active_sensor_count','hfai','hfai_method','raw_sensor_detections',
           'power_t2m_c','power_temp_anomaly_c','power_precip_mm_day','power_precip_mm_month','power_precip_anomaly_mm','power_precip_anomaly_pct','power_precip_z']
    monthly=monthly[[c for c in front if c in monthly.columns]].sort_values(['region','year','month','cell_lat_i','cell_lon_i'])
    annual_seam=calendar.loc[calendar.year.isin([2011,2012])].groupby(['region','year'],as_index=False).agg(
        hfai=('hfai','sum'),raw_sensor_detections=('raw_sensor_detections','sum'))
    seam_summary=[]
    for region in REGIONS:
        rows=annual_seam[annual_seam.region.eq(region)].set_index('year')
        if 2011 in rows.index and 2012 in rows.index:
            before=rows.loc[2011]; after=rows.loc[2012]
            seam_summary.append({'region':region,'raw_2011':float(before.raw_sensor_detections),'raw_2012':float(after.raw_sensor_detections),
                'raw_change_pct':float((after.raw_sensor_detections/before.raw_sensor_detections-1)*100) if before.raw_sensor_detections else None,
                'hfai_2011':float(before.hfai),'hfai_2012':float(after.hfai),
                'hfai_change_pct':float((after.hfai/before.hfai-1)*100) if before.hfai else None})
    monthly_path=OUTPUT_DIR/'harmonized_grid_025_monthly.csv.gz'
    monthly.to_csv(monthly_path,index=False,compression='gzip',float_format='%.5f')
    calendar['month_start']=pd.to_datetime(dict(year=calendar.year,month=calendar.month,day=1))
    calendar.to_csv(OUTPUT_DIR/'harmonized_calendar.csv',index=False,float_format='%.5f')
    validation_report={'project':'Pyrochron','index':'Harmonized Fire Activity Index (HFAI)','index_unit':'estimated MODIS-equivalent active-fire detections per 0.25-degree cell-month',
      'cell_degrees':CELL_DEGREES,'train_years':list(TRAIN_YEARS),'holdout_years':list(TEST_YEARS),'selected_method':selected,
      'method_selection':{'selected_method':selected,'selection_metric':'pooled MAE','fit_years':list(METHOD_FIT_YEARS),
                          'selection_years':list(METHOD_SELECTION_YEARS),'selection_scores':method_selection_scores,
                          'final_holdout_used_for_selection':False,
                          'note':'Candidate factors are fit on 2012-2017; method selection uses 2018-2019; final calibration factors are refit on 2012-2019; 2020-2024 is reserved for final out-of-time evaluation.'},
      'sensor_mission_starts':SENSOR_START,
      'historical_coverage_note':'Only the four named FIRMS request rectangles; the build is extensible to other AOIs when source archives are supplied.',
      'quality_rules':{'archive_type':'type = 0 when type field exists','MODIS_confidence':'numeric >= 30','VIIRS_confidence':'categorical n or h','NRT_without_type':'type filter not applied; retained after strict archive cutoff and marked provisional'},
      'candidate_scores':sorted(validation,key=lambda x:x['pooled_holdout_mae']),
      'candidate_scores_note':'2020-2024 is the final holdout. The selected method was chosen independently on 2018-2019; cross-candidate holdout rankings are descriptive and were not used for selection.',
      'sensor_metrics':calibration_rows,'calibration_factors':factors,
      'source_audit':file_audit,'source_files_expected':16,'source_files_archived':len([x for x in file_audit if x['source_quality']=='archive']),
      'seam_2011_2012':seam_summary,
      'seam_interpretation':'Descriptive annual changes across the S-NPP start; fire activity also changes between years, so the adjusted difference is not an isolated causal estimate of sensor seam removal.',
      'missing_sources':[],
      'available_archive_sensors_by_region':{r:sorted({x['sensor'] for x in file_audit if x['region']==r and x['source_quality']=='archive'}) for r in REGIONS},
      'limitations':['HFAI counts active-fire detections, not unique fires, burned area, or fire intensity.',
                     'MODIS confidence >=30 and VIIRS nominal/high are operational filters with different sensor semantics, not identical probability thresholds.',
                     'NOAA-21 has no 2012–2019 training overlap; VIIRS C2 family calibration is transferred from S-NPP/NOAA-20 and checked in the 2020–2024 holdout.',
                     'The 0.25-degree grid, approximate 1-km cluster candidate, and monthly time step smooth local fire spread and cannot recover small/cool fires missed by earlier MODIS.',
                     'HFAI is defined as monthly activity in the FIRMS detection record and aggregate count alignment; it is not a conditional detection rate or event-cause attribution.',
                     'NRT is included only after the standard archive date cutoff; recent rows are provisional and can be revised.',
                     'NASA POWER weather is centroid MERRA-2 assimilation output and is contextual association only; it does not establish causation.']}
    report_path=OUTPUT_DIR/'validation_report.json'
    report_path.write_text(json.dumps(validation_report,indent=2,allow_nan=False),encoding='utf-8')
    summary=calendar.groupby('region').agg(first_year=('year','min'),last_year=('year','max'),months=('month','count'),hfai_total=('hfai','sum'),raw_total=('raw_sensor_detections','sum')).reset_index()
    print('Selected method:',selected)
    print('Candidate scores:',json.dumps(validation_report['candidate_scores'],indent=2))
    print('Regions:',summary.to_string(index=False))
    print('Grid rows:',len(monthly),'calendar rows:',len(calendar),'gzip MB:',round(monthly_path.stat().st_size/1e6,2))
    print('Archive sources:',validation_report['source_files_archived'],'/',validation_report['source_files_expected'])
    print('Saved:',monthly_path,OUTPUT_DIR/'harmonized_calendar.csv',report_path)
    return validation_report

def selected_field(method: str) -> str:
    return {'count_ratio':'count','cluster_collapse':'cluster_count','frp_proxy':'frp_mw'}[method]

if __name__=='__main__':
    build()
