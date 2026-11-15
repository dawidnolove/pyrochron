from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd

import traceback

ROOT=Path(__file__).resolve().parent
def _load_local_env(path):
    if not path.exists(): return
    for line in path.read_text(encoding='utf-8').splitlines():
        line=line.strip()
        if not line or line.startswith('#') or '=' not in line: continue
        name,value=line.split('=',1)
        os.environ.setdefault(name.strip(),value.strip().strip('"').strip("'"))
_load_local_env(ROOT/'.env')
sys.path.insert(0,str(ROOT/'src'))
from pyrochron.analytics import DEFAULT_ACTIVITY_MEASURE,REGIONS,apply_activity_measure,load_grid,load_report
from pyrochron.realtime import query_roi

HOST=os.environ.get('PYROCHRON_HOST','127.0.0.1')
PORT=int(os.environ.get('PORT','8765'))
GRID=load_grid()
REPORT=load_report()
_MEASURED_GRIDS={}

def _grid_for_measure(measure):
    if measure not in _MEASURED_GRIDS:
        _MEASURED_GRIDS[measure]=apply_activity_measure(GRID,REPORT,measure)
    return _MEASURED_GRIDS[measure]


def _box_intersects(box, region_box):
    w,s,e,n=box; rw,rs,re,rn=region_box
    return w<re and e>rw and s<rn and n>rs

def _json_safe(value):
    if value is None: return None
    if isinstance(value,(np.integer,)): return int(value)
    if isinstance(value,(np.floating,float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value,(pd.Timestamp,datetime)): return value.isoformat()
    return value

def calendar_query(params):
    if 'bbox' not in params or 'from' not in params or 'to' not in params:
        raise ValueError('Required query parameters: bbox=west,south,east,north, from=YYYY-MM-DD, to=YYYY-MM-DD')
    try: box=tuple(float(x) for x in params['bbox'][0].split(','))
    except Exception as exc: raise ValueError('bbox must contain four comma-separated numbers: west,south,east,north') from exc
    if len(box)!=4: raise ValueError('bbox must contain west,south,east,north')
    west,south,east,north=box
    if not (-180<=west<east<=180 and -90<=south<north<=90):
        raise ValueError('bbox coordinates are out of range or not ordered west,south,east,north')
    if not all(re.fullmatch(r'\d{4}-\d{2}-\d{2}', params.get(k,[''])[0]) for k in ('from','to')):
        raise ValueError('from/to must be valid ISO dates such as 2020-01-01')
    try:
        start=pd.Timestamp(params['from'][0]).to_period('M').start_time
        end=pd.Timestamp(params['to'][0]).to_period('M').start_time
    except Exception as exc: raise ValueError('from/to must be valid ISO dates such as 2020-01-01') from exc
    if end<start: raise ValueError('to must be on or after from')
    chosen=params.get('region',[None])[0]
    regions=[chosen] if chosen else [r for r,spec in REGIONS.items() if _box_intersects(box,spec['bbox'])]
    if any(r not in REGIONS for r in regions): raise ValueError('region must be one of: '+', '.join(REGIONS))
    requested_measure=params.get('measure',[DEFAULT_ACTIVITY_MEASURE])[0].strip().lower()
    measure_aliases={'count_ratio':'count_aligned_hfai','count_aligned':'count_aligned_hfai',
                     'count_aligned_hfai':'count_aligned_hfai','cluster_collapse':'cluster_collapse'}
    if requested_measure not in measure_aliases:
        raise ValueError('measure must be count_aligned_hfai or cluster_collapse')
    measure=measure_aliases[requested_measure]
    selected_method='cluster_collapse' if measure=='cluster_collapse' else REPORT.get('selected_method','count_ratio')
    measured_grid=_grid_for_measure(measure)
    latlo=int(np.floor(south*4)); lathi=int(np.ceil(north*4)-1)
    lonlo=int(np.floor(west*4)); lonhi=int(np.ceil(east*4)-1)
    spatial=measured_grid.loc[measured_grid.region.isin(regions)&measured_grid.cell_lat_i.between(latlo,lathi)&measured_grid.cell_lon_i.between(lonlo,lonhi)].copy()
    part=spatial.loc[spatial.month_start.between(start,end)].copy()
    if part.empty:
        return {'meta':{'index':REPORT['index'],'unit':REPORT['index_unit'],'measure':measure,'default_measure':DEFAULT_ACTIVITY_MEASURE,'bbox':box,'from':start.date().isoformat(),'to':end.date().isoformat(),'regions':regions,
                        'selected_method':selected_method,'coverage':'The request does not intersect precomputed observations in the configured AOIs.'},'data':[]}
    agg={'hfai':'sum','count_aligned_hfai':'sum','cluster_hfai':'sum','raw_sensor_detections':'sum','MODIS_count':'sum','SNPP_VIIRS_count':'sum','NOAA20_VIIRS_count':'sum','NOAA21_VIIRS_count':'sum','active_sensor_count':'max'}
    for sensor in ('MODIS','SNPP_VIIRS','NOAA20_VIIRS','NOAA21_VIIRS'):
        for light in ('day','night'):
            col=f'{sensor}_{light}_count'
            if col in part: agg[col]='sum'
    for col in ('MODIS_Terra_count','MODIS_Aqua_count','MODIS_Terra_day_count','MODIS_Terra_night_count','MODIS_Aqua_day_count','MODIS_Aqua_night_count'):
        if col in part: agg[col]='sum'
    for c in ['power_t2m_c','power_temp_anomaly_c','power_precip_mm_month','power_precip_anomaly_mm','power_precip_anomaly_pct','power_precip_z']:
        if c in part: agg[c]='mean'
    monthly=part.groupby(['region','year','month'],as_index=False).agg(agg)
    baseline=spatial[spatial.year.between(2012,2019)].groupby(['region','year','month'],as_index=False).hfai.sum()
    if not baseline.empty:
        med=baseline.groupby(['region','month']).hfai.median().rename('hfai_month_median').reset_index()
        monthly=monthly.merge(med,on=['region','month'],how='left')
    else: monthly['hfai_month_median']=np.nan
    monthly['hfai_vs_median_ratio']=np.where(monthly.hfai_month_median>0,monthly.hfai/monthly.hfai_month_median,np.nan)
    monthly['hfai_anomaly_pct']=np.where(monthly.hfai_month_median>0,100*(monthly.hfai-monthly.hfai_month_median)/monthly.hfai_month_median,np.nan)
    monthly['month_start']=pd.to_datetime(dict(year=monthly.year,month=monthly.month,day=1))
    monthly=monthly.sort_values(['region','year','month'])
    return {'meta':{'index':REPORT['index'],'unit':REPORT['index_unit'],'measure':measure,'default_measure':DEFAULT_ACTIVITY_MEASURE,'resolution_degrees':0.25,'bbox':box,
                    'from':start.date().isoformat(),'to':end.date().isoformat(),'regions':regions,
                    'selected_method':selected_method,'training_years':REPORT['train_years'],'validation_years':REPORT['holdout_years'],
                    'baseline_years':[2012,2019],'source':'NASA FIRMS MODIS C6.1 + VIIRS Collection 2 archives; NASA POWER MERRA-2 context',
                    'coverage':'Historical data are available only inside configured AOI rectangles and for source archives present in the build.'},
            'data':[{k:_json_safe(v) for k,v in row.items()} for row in monthly.to_dict(orient='records')]}


def roi_query(params):
    """Live NRT endpoint. Its seven-day window is fixed in the server, not URL parameters."""
    if 'bbox' not in params:
        raise ValueError('Required query parameter: bbox=west,south,east,north')
    key=os.environ.get('FIRMS_MAP_KEY','').strip()
    if not re.fullmatch(r'[A-Za-z0-9]{32}',key):
        raise PermissionError('FIRMS_MAP_KEY is missing or invalid. Set it in the backend .env file.')
    try:
        bbox=tuple(float(v) for v in params['bbox'][0].split(','))
    except (TypeError,ValueError):
        raise ValueError('bbox must contain four numbers: west,south,east,north') from None
    try:
        return query_roi(key,bbox,REPORT)
    except ValueError:
        raise
    except RuntimeError as exc:
        raise ConnectionError(str(exc)) from None

class Handler(BaseHTTPRequestHandler):
    server_version='PyrochronAPI/1.0'
    def _write(self,status,body,content_type='application/json; charset=utf-8'):
        raw=body if isinstance(body,bytes) else body.encode('utf-8')
        self.send_response(status); self.send_header('Content-Type',content_type); self.send_header('Content-Length',str(len(raw)))
        self.send_header('Cache-Control','no-store'); self.end_headers(); self.wfile.write(raw)
    def do_GET(self):
        parsed=urlparse(self.path); params=parse_qs(parsed.query)
        if parsed.path in ('/healthz','/health'):
            self._write(200,json.dumps({'status':'ok','project':'Pyrochron','regions':list(REGIONS),'grid_rows':len(GRID)})); return
        if parsed.path in ('/','/docs'):
            doc={'service':'Pyrochron calendar API','endpoint':'/api/calendar?bbox=west,south,east,north&from=YYYY-MM-DD&to=YYYY-MM-DD&region=Quebec&measure=count_aligned_hfai|cluster_collapse',
                 'available_regions':{k:v['bbox'] for k,v in REGIONS.items()},'measures':['cluster_collapse','count_aligned_hfai'],'default_measure':DEFAULT_ACTIVITY_MEASURE,'example':'/api/calendar?bbox=-78,46,-64,55&from=2020-01-01&to=2024-12-31&region=Quebec&measure=cluster_collapse',
                 'limits':'The calendar route reads precomputed historical values limited to configured AOIs. The separate ROI route queries NASA FIRMS NRT and requires a server-side map key.'}
            doc['live_roi']={'endpoint':'/api/roi?bbox=west,south,east,north','window_days':7,'measure':'cluster_collapse','key':'Set FIRMS_MAP_KEY in the backend environment; the key is never sent to the browser.'}
            self._write(200,json.dumps(doc)); return
        if parsed.path=='/api/roi':
            try:
                result=roi_query(params)
                self._write(200,json.dumps(result,default=_json_safe,allow_nan=False,separators=(',',':')))
            except ValueError as exc:
                self._write(400,json.dumps({'error':str(exc)}))
            except PermissionError as exc:
                self._write(503,json.dumps({'error':str(exc)}))
            except ConnectionError as exc:
                self._write(502,json.dumps({'error':str(exc)}))
            except Exception as exc:
                print("\n========== NRT PROCESSING ERROR ==========")
                traceback.print_exc()
                print("==========================================\n")

                self._write(
                    500,
                    json.dumps({
                        'error': 'Internal NRT processing error',
                        'detail': str(exc),
                        'type': type(exc).__name__
                    })
                )
            # except Exception:
            #     self._write(500,json.dumps({'error':'Internal NRT processing error'}))
            return
        if parsed.path!='/api/calendar':
            self._write(404,json.dumps({'error':'Not found','available':['/healthz','/docs','/api/calendar','/api/roi']})); return
        try:
            result=calendar_query(params)
            if params.get('format',['json'])[0].lower()=='csv':
                data=pd.DataFrame(result['data'])
                self._write(200,data.to_csv(index=False),'text/csv; charset=utf-8'); return
            self._write(200,json.dumps(result,allow_nan=False,separators=(',',':')))
        except ValueError as exc:
            self._write(400,json.dumps({'error':str(exc)}))
        except Exception as exc:
            self._write(500,json.dumps({'error':'Internal processing error','detail':str(exc)}))
    def log_message(self,fmt,*args):
        print(f'{self.log_date_time_string()} {self.address_string()} {fmt%args}')

if __name__=='__main__':
    print(f'Pyrochron API listening on http://{HOST}:{PORT}; loaded {len(GRID):,} grid rows.')
    ThreadingHTTPServer((HOST,PORT),Handler).serve_forever()
