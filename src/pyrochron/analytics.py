from __future__ import annotations

import gzip
import json
import math
import re
from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .pipeline import (CELL_DEGREES, CODE_SENSOR, REGIONS, REQUEST_REGION, SENSOR_LABEL,
                       SENSOR_START, SENSORS, _confidence_mask, _file_identity, _prepare_frame)

ROOT=Path(__file__).resolve().parents[2]
DATA_DIR=ROOT/'outputs'/'data'
ARCHIVE_DIR=ROOT/'archives'
GRID_PATH=DATA_DIR/'harmonized_grid_025_monthly.csv.gz'
CALENDAR_PATH=DATA_DIR/'harmonized_calendar.csv'
REPORT_PATH=DATA_DIR/'validation_report.json'
POINT_EXPORT_MAX_ROWS=300_000
POINT_EXPORT_CHUNK_ROWS=50_000
POINT_EXPORT_METHOD=('Two-pass Hamilton largest-remainder quotas by sensor, then month; '
                     'lowest deterministic pandas row hash within each sensor-month stratum; '
                     'final output ordered by sensor, month, and row hash.')


def load_calendar() -> pd.DataFrame:
    df=pd.read_csv(CALENDAR_PATH,parse_dates=['month_start'])
    return df

def load_grid() -> pd.DataFrame:
    df=pd.read_csv(GRID_PATH,parse_dates=['month_start'])
    return df

def load_report() -> dict[str,Any]:
    return json.loads(REPORT_PATH.read_text(encoding='utf-8'))

DEFAULT_ACTIVITY_MEASURE='cluster_collapse'

def apply_activity_measure(grid: pd.DataFrame, report: dict[str,Any], measure: str=DEFAULT_ACTIVITY_MEASURE,
                           available_sensors_by_region: dict[str,set[str]] | None=None) -> pd.DataFrame:
    """Attach both supported measures and select one as ``hfai`` for downstream use."""
    aliases={'count_ratio':'count_aligned_hfai','count_aligned':'count_aligned_hfai',
             'count_aligned_hfai':'count_aligned_hfai','cluster_collapse':'cluster_collapse'}
    if measure not in aliases:
        raise ValueError('measure must be count_aligned_hfai or cluster_collapse')
    chosen=aliases[measure]
    result=grid.copy()
    result['count_aligned_hfai']=pd.to_numeric(result['hfai'],errors='coerce')
    result['cluster_hfai']=np.nan
    for region, rows in result.groupby('region',sort=False):
        factors=report.get('calibration_factors',{}).get(region,{})
        dates=pd.to_datetime(dict(year=rows.year,month=rows.month,day=1))
        available=(available_sensors_by_region or {}).get(region)
        estimates=[]; active=[]
        if available is None or 'MODIS' in available:
            estimates.append(pd.to_numeric(rows['MODIS_count'],errors='coerce').fillna(0.0).rename('MODIS'))
            active.append(dates.ge(pd.Timestamp(SENSOR_START['MODIS']).to_period('M').start_time).rename('MODIS'))
        for sensor in SENSORS:
            if sensor=='MODIS':
                continue
            if available is not None and sensor not in available:
                continue
            factor=factors.get(sensor,{}).get('cluster_ratio')
            source=f'{sensor}_cluster_count'
            if factor is None or not np.isfinite(factor) or source not in rows:
                continue
            estimates.append((pd.to_numeric(rows[source],errors='coerce').fillna(0.0)*float(factor)).rename(sensor))
            active.append(dates.ge(pd.Timestamp(SENSOR_START[sensor]).to_period('M').start_time).rename(sensor))
        aligned=pd.concat(estimates,axis=1)
        availability=pd.concat(active,axis=1).set_axis(aligned.columns,axis=1)
        result.loc[rows.index,'cluster_hfai']=aligned.where(availability.to_numpy()).mean(axis=1,skipna=True).to_numpy()
    result['hfai']=result['cluster_hfai'] if chosen=='cluster_collapse' else result['count_aligned_hfai']
    result['hfai_method']='cluster_collapse' if chosen=='cluster_collapse' else 'count_ratio'
    return result

def box_for_region(region: str) -> tuple[float,float,float,float]:
    west,south,east,north=REGIONS[region]['bbox']
    return west,south,east,north

def select_grid(grid: pd.DataFrame, region: str, bbox: tuple[float,float,float,float], start: str, end: str) -> pd.DataFrame:
    west,south,east,north=bbox
    if west>=east or south>=north:
        return grid.iloc[0:0].copy()
    # A grid cell is included if its 0.25-degree footprint intersects the requested bbox.
    lon_lo=math.floor(west/CELL_DEGREES)
    lon_hi=math.ceil(east/CELL_DEGREES)-1
    lat_lo=math.floor(south/CELL_DEGREES)
    lat_hi=math.ceil(north/CELL_DEGREES)-1
    begin=pd.Timestamp(start).to_period('M').start_time
    finish=pd.Timestamp(end).to_period('M').start_time
    mask=(grid.region.eq(region)&grid.cell_lon_i.between(lon_lo,lon_hi)&grid.cell_lat_i.between(lat_lo,lat_hi)&
          grid.month_start.between(begin,finish))
    return grid.loc[mask].copy()

def select_calendar(calendar: pd.DataFrame, region: str, start: str, end: str) -> pd.DataFrame:
    begin=pd.Timestamp(start).to_period('M').start_time
    finish=pd.Timestamp(end).to_period('M').start_time
    return calendar.loc[calendar.region.eq(region)&calendar.month_start.between(begin,finish)].copy()

def empirical_percentile(value: float, reference: Any) -> float | None:
    """Empirical percentile rank (share of baseline observations <= value)."""
    values=pd.to_numeric(pd.Series(reference),errors='coerce').to_numpy(dtype=float)
    values=values[np.isfinite(values)]
    if not np.isfinite(value) or not len(values): return None
    return float(100*np.count_nonzero(values<=float(value))/len(values))

def make_calendar_png(calendar: pd.DataFrame, region: str, start: str, end: str,
                      index_label: str='HFAI') -> bytes:
    data=select_calendar(calendar,region,start,end)
    if data.empty:
        data=calendar.loc[calendar.region.eq(region)].copy()
    pivot=data.pivot(index='year',columns='month',values='hfai').reindex(columns=range(1,13))
    years=pivot.index.to_numpy()
    fig_h=max(3.2,min(9.0,1.3+len(years)*0.22))
    paper='#f5f4ef'; ink='#202723'; muted='#515b55'; line='#cbd2ca'
    cmap=matplotlib.colors.LinearSegmentedColormap.from_list('pyrochron_activity',['#829985','#718b77','#5c7b64','#315a49'])
    fig,ax=plt.subplots(figsize=(10.2,fig_h),facecolor=paper)
    ax.set_facecolor(paper)
    vals=pivot.to_numpy(dtype=float)
    im=ax.imshow(vals,aspect='auto',cmap=cmap,interpolation='nearest')
    ax.set_xticks(range(12),['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    ax.set_yticks(range(len(years)),years)
    ax.set_xlabel('Month',labelpad=8,color=ink)
    ax.set_ylabel('Year',labelpad=8,color=ink)
    ax.set_title(f'{REGIONS[region]["label"]} · {index_label}',loc='left',pad=14,
                 fontsize=15,fontfamily='DejaVu Serif',color='#222925')
    ax.tick_params(length=0,labelsize=8,colors=muted)
    ax.set_xticks(np.arange(-.5,12,1),minor=True); ax.set_yticks(np.arange(-.5,len(years),1),minor=True)
    ax.grid(which='minor',color=paper,linewidth=1.2); ax.tick_params(which='minor',bottom=False,left=False)
    cb=fig.colorbar(im,ax=ax,pad=.02); cb.set_label('MODIS-equivalent detections per region-month',color=ink); cb.outline.set_visible(False)
    fig.text(.01,.01,f'NASA FIRMS active-fire detections · 0.25° monthly grid · {index_label} is not a unique-fire count or burned area',fontsize=8,color=muted)
    fig.tight_layout(rect=(0,.03,1,1))
    out=BytesIO(); fig.savefig(out,format='png',dpi=180,bbox_inches='tight',facecolor=fig.get_facecolor()); plt.close(fig)
    return out.getvalue()

def _archive_cutoffs(report: dict[str,Any]) -> dict[tuple[str,str],pd.Timestamp]:
    out={}
    for item in report.get('source_audit',[]):
        if item.get('source_quality')=='archive':
            key=(item['region'],item['sensor']); date=pd.Timestamp(item['date_max'])
            out[key]=max(out.get(key,date),date)
    return out

def _source_end_dates(report: dict[str,Any]) -> dict[tuple[str,str],pd.Timestamp]:
    out={}
    for item in report.get('source_audit',[]):
        if item.get('date_max'):
            key=(item['region'],item['sensor']); date=pd.Timestamp(item['date_max'])
            out[key]=max(out.get(key,date),date)
    return out

def _allocate_proportional_quotas(counts: dict[Any,int], cap: int, ensure_one: bool=False) -> dict[Any,int]:
    """Hamilton largest-remainder allocation, bounded by each stratum's population."""
    keys=sorted(k for k,v in counts.items() if int(v)>0)
    total=sum(int(counts[k]) for k in keys)
    cap=max(0,min(int(cap),total))
    if not total or not cap: return {k:0 for k in counts}
    exact={k:cap*int(counts[k])/total for k in keys}
    quotas={k:min(int(counts[k]),int(math.floor(exact[k]))) for k in keys}
    if ensure_one and cap>=len(keys):
        for k in keys:
            if quotas[k]==0: quotas[k]=1
    # Correct any seats consumed by the minimum-one rule before distributing remainders.
    while sum(quotas.values())>cap:
        candidates=[k for k in keys if quotas[k]>(1 if ensure_one and cap>=len(keys) else 0)]
        if not candidates: break
        k=min(candidates,key=lambda x:(exact[x]-math.floor(exact[x]),-quotas[x],str(x)))
        quotas[k]-=1
    while sum(quotas.values())<cap:
        candidates=[k for k in keys if quotas[k]<int(counts[k])]
        if not candidates: break
        k=max(candidates,key=lambda x:(exact[x]-math.floor(exact[x]),int(counts[x])-quotas[x],str(x)))
        quotas[k]+=1
    return {k:quotas.get(k,0) for k in counts}

def _allocate_sensor_month_quotas(stratum_counts: dict[tuple[str,str],int], cap: int) -> dict[tuple[str,str],int]:
    by_sensor: dict[str,int]={}
    for (sensor,_month),count in stratum_counts.items(): by_sensor[sensor]=by_sensor.get(sensor,0)+int(count)
    cap=max(0,min(int(cap),sum(by_sensor.values())))
    sensor_quota=_allocate_proportional_quotas(by_sensor,cap,ensure_one=True)
    result={}
    for sensor,total in by_sensor.items():
        months={month:int(count) for (src,month),count in stratum_counts.items() if src==sensor}
        month_quota=_allocate_proportional_quotas(months,sensor_quota.get(sensor,0))
        result.update({(sensor,month):quota for month,quota in month_quota.items()})
    return result

def _keep_lowest_priority(current: pd.DataFrame, candidates: pd.DataFrame, quota: int) -> pd.DataFrame:
    if quota<=0: return candidates.iloc[0:0].copy()
    if current.empty: combined=candidates
    elif candidates.empty: combined=current
    else: combined=pd.concat([current,candidates],ignore_index=True)
    if len(combined)>quota:
        combined=combined.nsmallest(quota,'_sample_priority',keep='first')
    return combined

def _point_priority(frame: pd.DataFrame) -> pd.Series:
    identity=[c for c in ('sensor','acq_date','acq_time','latitude','longitude','satellite','instrument','version',
                           'brightness','bright_t31','frp','scan','track','confidence_raw','daynight','type') if c in frame]
    return pd.util.hash_pandas_object(frame[identity].astype(str),index=False).astype('uint64')

def export_hotspots(region: str, bbox: tuple[float,float,float,float], start: str, end: str,
                    report: dict[str,Any], max_rows: int=POINT_EXPORT_MAX_ROWS,
                    months: list[int] | None=None) -> tuple[bytes,dict[str,Any]]:
    """Export all matches or a deterministic sensor/month-proportional capped sample."""
    west,south,east,north=bbox
    start_ts=pd.Timestamp(start).normalize(); end_ts=pd.Timestamp(end).normalize()+pd.Timedelta(days=1)-pd.Timedelta(microseconds=1)
    max_rows=max(0,int(max_rows))
    selected_months=sorted({int(month) for month in months if 1<=int(month)<=12}) if months else list(range(1,13))
    cutoff_map=_archive_cutoffs(report); end_map=_source_end_dates(report)
    method=report.get('selected_method','count_ratio')
    factors=report.get('calibration_factors',{}).get(region,{})
    available=set(report.get('available_archive_sensors_by_region',{}).get(region,[]))
    paths=[]
    for path in ARCHIVE_DIR.glob('fire_*.csv'):
        ident=_file_identity(path)
        if ident is None: continue
        collection,code,rid=ident
        sensor=CODE_SENSOR.get(code)
        if REQUEST_REGION[rid]==region and sensor in available:
            paths.append((path,collection,sensor))
    paths.sort(key=lambda item:item[0].name.casefold())

    def matching_chunks(path: Path, collection: str, sensor: str):
        cutoff=cutoff_map.get((region,sensor))
        for raw in pd.read_csv(path,chunksize=POINT_EXPORT_CHUNK_ROWS,low_memory=False):
            frame=_prepare_frame(raw,sensor)
            if frame.empty: continue
            frame['region']=region
            mask=(frame.latitude.between(south,north)&frame.longitude.between(west,east)&frame.acq_date.between(start_ts,end_ts))
            mask &= frame.acq_date.dt.month.isin(selected_months)
            if collection=='nrt' and cutoff is not None: mask &= frame.acq_date.gt(cutoff)
            frame=frame.loc[mask].copy()
            if not frame.empty: yield frame

    # Pass one counts every eligible record, even if the final file will be capped.
    matched_strata: dict[tuple[str,str],int]={}
    for path,collection,sensor in paths:
        for frame in matching_chunks(path,collection,sensor):
            month_counts=frame.acq_date.dt.to_period('M').astype(str).value_counts()
            for month,count in month_counts.items():
                key=(sensor,str(month)); matched_strata[key]=matched_strata.get(key,0)+int(count)
    matched_by_sensor={sensor:sum(n for (src,_),n in matched_strata.items() if src==sensor) for sensor in SENSORS}
    matched_total=sum(matched_by_sensor.values())
    quotas=_allocate_sensor_month_quotas(matched_strata,max_rows)

    # Pass two keeps only the lowest stable row hashes within each sensor/month quota.
    reservoirs: dict[tuple[str,str],pd.DataFrame]={}
    wanted=['region','sensor_name','acq_date','acq_time','latitude','longitude','brightness','bright_t31','frp','scan','track','confidence_raw','daynight','satellite','instrument','version','type','type_filter_applied','hfai_source_weight','selected_harmonization','archive_quality']
    for path,collection,sensor in paths:
        for frame in matching_chunks(path,collection,sensor):
            factor=1.0 if sensor=='MODIS' else factors.get(sensor,{}).get({'count_ratio':'count_ratio','cluster_collapse':'cluster_ratio','frp_proxy':'frp_to_modis_count'}.get(method,'count_ratio'))
            if factor is None or not np.isfinite(factor): factor=1.0
            months=frame.acq_date.dt.to_period('M')
            denominators=months.map(lambda month:max(1,sum(1 for s in available if
                pd.Timestamp(SENSOR_START[s]).to_period('M')<=month and
                end_map.get((region,s),pd.Timestamp(SENSOR_START[s])).to_period('M')>=month))).to_numpy()
            frame['hfai_source_weight']=float(factor)/denominators
            if method=='frp_proxy' and sensor!='MODIS':
                frame['hfai_source_weight']=pd.to_numeric(frame['frp'],errors='coerce').fillna(0)*float(factor)/denominators
            if method=='cluster_collapse' and sensor!='MODIS':
                y=np.floor(frame.latitude*110.574).astype('int64'); x=np.floor(frame.longitude*111.32*np.cos(np.deg2rad(frame.latitude))).astype('int64')
                keys=pd.DataFrame({'date':frame.acq_date,'daynight':frame.daynight,'x':x,'y':y},index=frame.index)
                sizes=keys.groupby(['date','daynight','x','y'])['x'].transform('size').to_numpy()
                frame['hfai_source_weight']=float(factor)/np.maximum(sizes,1)/denominators
            frame['sensor']=sensor; frame['sensor_name']=SENSOR_LABEL[sensor]
            frame['archive_quality']='archive' if collection=='archive' else 'NRT after standard-archive cutoff'
            frame['type_filter_applied']=bool('type' in frame.columns)
            frame['selected_harmonization']=method
            frame['_sample_month']=frame.acq_date.dt.to_period('M').astype(str)
            frame['_sample_priority']=_point_priority(frame)
            for month,candidates in frame.groupby('_sample_month',sort=False):
                key=(sensor,str(month)); quota=quotas.get(key,0)
                selected=candidates[[c for c in wanted if c in candidates]+['_sample_month','_sample_priority']].copy()
                reservoirs[key]=_keep_lowest_priority(reservoirs.get(key,selected.iloc[0:0]),selected,quota)
    sampled=[]
    for key in sorted(reservoirs):
        part=reservoirs[key]
        if not part.empty: sampled.append(part)
    out=pd.concat(sampled,ignore_index=True) if sampled else pd.DataFrame(columns=wanted+['_sample_priority'])
    if not out.empty:
        out=out.sort_values(['sensor_name','_sample_month','_sample_priority'],kind='mergesort')
    exported_by_sensor={sensor:0 for sensor in SENSORS}
    if not out.empty:
        for label,count in out.sensor_name.value_counts().items():
            sensor=next((s for s,name in SENSOR_LABEL.items() if name==label),None)
            if sensor: exported_by_sensor[sensor]=int(count)
    exported_total=len(out)
    truncated=matched_total>exported_total
    per_sensor={sensor:{'matched':int(matched_by_sensor[sensor]),'exported':int(exported_by_sensor[sensor])} for sensor in SENSORS}
    per_sensor_month={}
    for key,count in sorted(matched_strata.items()):
        per_sensor_month[f'{key[0]}:{key[1]}']={'matched':int(count),'exported':int(quotas.get(key,0))}
    manifest={'project':'Pyrochron','region':region,'bbox':[float(west),float(south),float(east),float(north)],
        'from':start_ts.date().isoformat(),'to':pd.Timestamp(end).date().isoformat(),
        'months':selected_months,
        'quality_rules':{'archive_type':'FIRMS type 0 when the type field exists','MODIS_confidence':'numeric confidence >= 30',
                         'VIIRS_confidence':'categorical confidence n or h','NRT_without_type':'type filter is unavailable; such rows are retained only strictly after the matching standard archive cutoff and marked provisional'},
        'matched_rows':int(matched_total),'exported_rows':int(exported_total),'row_cap':max_rows,'truncated':bool(truncated),
        'is_sample':bool(truncated),'sampling_method':POINT_EXPORT_METHOD if truncated else 'No sampling; all matched rows are included.',
        'per_sensor':per_sensor,'per_sensor_month':per_sensor_month}
    out=out[[c for c in wanted if c in out.columns]]
    compressed=gzip.compress(out.to_csv(index=False).encode('utf-8'),compresslevel=6,mtime=0)
    return compressed,manifest

def make_brief_pdf(region: str, start: str, end: str, summary: dict[str,Any], validation: dict[str,Any],
                   index_label: str='HFAI') -> bytes:
    buf=BytesIO()
    doc=SimpleDocTemplate(buf,pagesize=letter,rightMargin=.65*inch,leftMargin=.65*inch,topMargin=.56*inch,bottomMargin=.5*inch)
    styles=getSampleStyleSheet()
    title=ParagraphStyle('PyroTitle',parent=styles['Title'],fontName='Times-Bold',fontSize=22,leading=25,textColor=colors.HexColor('#202723'),alignment=TA_LEFT,spaceAfter=6)
    deck=ParagraphStyle('PyroDeck',parent=styles['BodyText'],fontName='Helvetica',fontSize=10,leading=14,textColor=colors.HexColor('#515b55'),spaceAfter=12)
    body=ParagraphStyle('PyroBody',parent=styles['BodyText'],fontName='Helvetica',fontSize=9.3,leading=13,textColor=colors.HexColor('#29322d'),spaceAfter=7)
    small=ParagraphStyle('PyroSmall',parent=body,fontSize=7.5,leading=10,textColor=colors.HexColor('#69726d'))
    area=REGIONS[region]
    hist=float(summary.get('hfai',0) or 0); raw=float(summary.get('raw_sensor_detections',0) or 0)
    med=summary.get('hfai_month_median'); ratio=summary.get('hfai_vs_median_ratio')
    story=[Paragraph('Pyrochron · field brief',title),Paragraph(f'{area["label"]} &nbsp; / &nbsp; {start} to {end}',deck)]
    stats=[[index_label,'Raw FIRMS records','0.25° cells','Sensors represented'],
           [f'{hist:,.1f}',f'{raw:,.0f}',f'{int(summary.get("grid_cells",0)):,}',str(summary.get('active_sensor_count','—'))]]
    table=Table(stats,colWidths=[1.65*inch]*4,rowHeights=[.26*inch,.38*inch])
    table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#d7e0d8')),('TEXTCOLOR',(0,0),(-1,-1),colors.HexColor('#26312b')),
      ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,0),8),('FONTNAME',(0,1),(-1,1),'Helvetica-Bold'),('FONTSIZE',(0,1),(-1,1),13),
      ('LEFTPADDING',(0,0),(-1,-1),8),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LINEBELOW',(0,0),(-1,0),.4,colors.HexColor('#a8b4aa'))]))
    story += [table,Spacer(1,12)]
    if med is not None and pd.notna(med):
        ratio_text=f'{float(ratio):.2f}× the same-month baseline' if ratio is not None and pd.notna(ratio) else 'baseline ratio unavailable because the median is zero'
        story.append(Paragraph(f'The selected period records {ratio_text}. The reference is the 2012–2019 median for the same calendar month, computed from the harmonized regional series.',body))
    story.append(Paragraph(f'What {index_label} means',ParagraphStyle('sub',parent=body,fontName='Helvetica-Bold',fontSize=11,spaceAfter=4)))
    story.append(Paragraph(f'{index_label} summarizes active-fire detections on a common 0.25° monthly grid using the selected view. VIIRS calibration factors are learned from 2012–2019 overlap data; candidate methods are compared on held-out 2020–2024 months. A reported value is a detection-based estimate, not a count of unique fires, burned area, or a measure of damage.',body))
    story.append(Paragraph('Reading the signal with care',ParagraphStyle('sub2',parent=body,fontName='Helvetica-Bold',fontSize=11,spaceAfter=4)))
    story.append(Paragraph('The measure summarizes monthly activity in the NASA FIRMS detection record and compares aggregate counts across sensors. Confidence fields have instrument-specific meanings. NOAA-21 uses a disclosed transfer from earlier VIIRS Collection 2 sensors because it did not overlap the 2012–2019 training period. The most recent records are provisional near-real-time detections. NASA POWER precipitation and temperature are centroid MERRA-2 context, not evidence of causation.',body))
    scores=validation.get('candidate_scores',[])
    if scores:
        chosen='cluster_collapse' if index_label=='Daily cluster proxy' else validation.get('selected_method','—')
        score=next((x for x in scores if x['method']==chosen),None)
        if score:
            story.append(Paragraph(f'Out-of-time candidate diagnostic · {chosen.replace("_"," ")}; pooled holdout MAE {score["pooled_holdout_mae"]:.2f}, RMSE {score["pooled_holdout_rmse"]:.2f} MODIS detections per cell-month. Candidate rankings on the holdout did not select the primary HFAI method.',body))
    bbox=', '.join(f'{v:g}' for v in area['bbox'])
    story.append(Spacer(1,5))
    story.append(Paragraph(f'Source: NASA FIRMS MODIS C6.1 and VIIRS Collection 2 archive exports; NASA POWER Monthly API (MERRA-2). AOI west/south/east/north: {bbox}. Grid cells: 0.25°. This is a monitoring aid, not a forecast or incident-level dispatch tool.',small))
    doc.build(story)
    return buf.getvalue()
