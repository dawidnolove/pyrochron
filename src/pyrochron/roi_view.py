from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import urlopen

import altair as alt
import folium
import pandas as pd
import streamlit as st
from folium.plugins import Draw, HeatMap
from streamlit_folium import st_folium


def _load_local_env():
    env_path=Path(__file__).resolve().parents[2]/'.env'
    if not env_path.exists(): return
    for line in env_path.read_text(encoding='utf-8').splitlines():
        line=line.strip()
        if not line or line.startswith('#') or '=' not in line: continue
        name,value=line.split('=',1)
        os.environ.setdefault(name.strip(),value.strip().strip('"').strip("'"))

_load_local_env()


def _carto_map(center, zoom):
    key=os.environ.get('CARTO_API_KEY','').strip()
    m=folium.Map(location=center,zoom_start=zoom,tiles=None,control_scale=True)
    if key:
        tile_url='https://{s}.basemaps.cartocdn.com/rastertiles/light_all/{z}/{x}/{y}{r}.png?key='+quote(key,safe='')
        folium.TileLayer(tiles=tile_url,attr='© OpenStreetMap contributors © CARTO',
                         name='CARTO Positron',subdomains='abcd',max_zoom=20).add_to(m)
    else:
        folium.TileLayer('OpenStreetMap',name='OpenStreetMap').add_to(m)
    return m


def _bounds(feature):
    if not feature:
        return None
    geometry=feature.get('geometry',{})
    coords=geometry.get('coordinates',[])
    points=[]
    def visit(value):
        if isinstance(value,(list,tuple)) and len(value)>=2 and all(isinstance(x,(int,float)) for x in value[:2]):
            points.append((float(value[0]),float(value[1])))
        elif isinstance(value,(list,tuple)):
            for child in value: visit(child)
    visit(coords)
    if not points: return None
    return (min(p[0] for p in points),min(p[1] for p in points),max(p[0] for p in points),max(p[1] for p in points))


def render_live_roi():
    st.markdown('## Live custom ROI · NRT last seven UTC days')
    st.write('Draw a rectangle or polygon on the map, or enter coordinates. NASA FIRMS Area API accepts bounding boxes; a polygon is queried using its enclosing bounding box.')
    m=_carto_map([20,0],2)
    Draw(export=False, draw_options={'polyline':False,'circle':False,'circlemarker':False,
          'marker':False,'polygon':{'allowIntersection':False},'rectangle':True},
          edit_options={'edit':True,'remove':True}).add_to(m)
    drawn=st_folium(m,height=430,use_container_width=True,key='live_roi_map',
                    returned_objects=['last_active_drawing'])
    feature=(drawn or {}).get('last_active_drawing')
    if not os.environ.get('CARTO_API_KEY','').strip():
        st.caption('CARTO_API_KEY is not configured; the map is using OpenStreetMap.')
    bounds=_bounds(feature)
    if bounds:
        for key,value in zip(('roi_west','roi_south','roi_east','roi_north'),bounds):
            st.session_state[key]=round(value,5)
        st.caption('Coordinates filled from the drawn feature’s enclosing bounding box.')
    cols=st.columns(4)
    defaults=(-78.0,46.0,-64.0,55.0)
    labels=('West longitude','South latitude','East longitude','North latitude')
    limits=((-180,180),(-90,90),(-180,180),(-90,90))
    values=[]
    for i, (col, label, key, default, bounds_limit) in enumerate(
        zip(
            cols,
            labels,
            ('roi_west', 'roi_south', 'roi_east', 'roi_north'),
            defaults,
            limits
        )
    ):
        min_val = float(bounds_limit[0])
        max_val = float(bounds_limit[1])

        # Initialize or correct session state BEFORE creating widget
        if key not in st.session_state:
            st.session_state[key] = float(default)

        current_value = float(st.session_state[key])
        st.session_state[key] = max(
            min_val,
            min(max_val, current_value)
        )

        with col:
            value = st.number_input(
                label,
                min_value=min_val,
                max_value=max_val,
                step=0.25,
                format="%.5f",
                key=key
            )

            values.append(value)

    west, south, east, north = values
    if st.button('Query NASA FIRMS · fixed 7-day window',type='primary',key='query_live_roi'):
        if west>=east or south>=north:
            st.error('Enter an ordered bbox: west < east and south < north.')
        else:
            api_base=os.environ.get('PYROCHRON_API_URL','http://127.0.0.1:8765').rstrip('/')
            bbox=','.join(f'{x:.6f}' for x in values)
            try:
                with st.spinner('Requesting four NRT sources and applying quality filters and daily cluster collapse…'):
                    request=urlopen(f'{api_base}/api/roi?bbox={quote(bbox,safe=",")}',timeout=150)
                    payload=json.loads(request.read())
                st.session_state['live_roi_result']=payload
                st.session_state['live_roi_error']=None
            except HTTPError as exc:
                try: message=json.loads(exc.read()).get('error','Request failed')
                except Exception: message=f'API returned HTTP {exc.code}'
                st.session_state['live_roi_error']=message
            except (URLError,TimeoutError,OSError) as exc:
                st.session_state['live_roi_error']=f'Cannot reach Pyrochron API ({type(exc).__name__}). Start api.py or use docker compose up.'
    if st.session_state.get('live_roi_error'):
        st.error(st.session_state['live_roi_error'])
    result=st.session_state.get('live_roi_result')
    if result:
        meta=result.get('meta',{})
        st.info(f"Live NASA FIRMS NRT · {meta.get('window_start_utc')} through {meta.get('window_end_utc')} UTC · fixed seven-day query · Cluster Collapse activity proxy.")
        if 'Factors are training-row-weighted' in meta.get('calibration_scope',''):
            st.warning(meta['calibration_scope'])
        states=result.get('source_status',{})
        st.caption('Source status: '+' · '.join(f'{name}: {state.get("status")}' + (f' ({state.get("message")})' if state.get('message') else '') for name,state in states.items()))
        totals=result.get('totals',{})
        a,b,c=st.columns(3)
        a.metric('Quality-filtered detections',f'{sum(totals.get(f"{s}_count",0) for s in ("MODIS","SNPP_VIIRS","NOAA20_VIIRS","NOAA21_VIIRS")):,}')
        b.metric('Daily cluster activity proxy',f'{totals.get("daily_cluster_proxy_sum",0):,.1f}')
        c.metric('Active 0.25° cells',f'{len(result.get("data",[])):,}')
        st.caption(meta.get('limitations',''))
        daily=pd.DataFrame(result.get('daily',[]))
        if not daily.empty:
            daily['date']=pd.to_datetime(daily['date'])
            st.markdown('### Daily fire-detection activity')
            st.caption('Approximate cluster-collapse activity by acquisition date across the selected ROI. This is a satellite-detection proxy, not a count of confirmed fire incidents.')
            chart=alt.Chart(daily).mark_bar(color='#315a49').encode(x=alt.X('date:T',title='Acquisition date (UTC)'),y=alt.Y('activity:Q',title='Daily activity proxy'),tooltip=[alt.Tooltip('date:T',title='Acquisition date (UTC)'),alt.Tooltip('activity:Q',title='Daily activity proxy',format=',.1f')]).properties(height=250,description='Daily approximate cluster-collapse activity proxy from NASA FIRMS NRT detections.')
            st.altair_chart(chart,use_container_width=True)
        points=result.get('data',[])
        if points:
            st.markdown('### Detection activity map')
            st.caption('Heat intensity shows the approximate cluster-collapse activity proxy per active 0.25° grid cell during the selected seven-day window. It does not show fire perimeters or incident locations.')
            heat=_carto_map([(south+north)/2,(west+east)/2],5)
            HeatMap([[p['latitude'],p['longitude'],p['hfai']] for p in points if p.get('hfai',0)>0],radius=13,blur=16).add_to(heat)
            st_folium(heat,height=440,use_container_width=True,key='live_roi_results',returned_objects=[])
            st.markdown('### Active grid-cell summary')
            st.caption('One row per active 0.25° grid cell. It lists the cell center, the cluster-collapse activity proxy, quality-filtered MODIS detections, and collapsed VIIRS clusters by sensor.')
            display_points=pd.DataFrame(points).sort_values('hfai',ascending=False).rename(columns={
                'latitude':'Cell center latitude', 'longitude':'Cell center longitude',
                'hfai':'Cluster-collapse activity proxy', 'MODIS_count':'MODIS filtered detections',
                'SNPP_VIIRS_cluster_count':'VIIRS S-NPP collapsed clusters',
                'NOAA20_VIIRS_cluster_count':'VIIRS NOAA-20 collapsed clusters',
                'NOAA21_VIIRS_cluster_count':'VIIRS NOAA-21 collapsed clusters'})
            st.dataframe(display_points,use_container_width=True,hide_index=True)
        else:
            st.success('The available sources returned no detections that passed the configured quality filters for this bbox and seven-day window.')
    st.caption('NRT source detections are provisional observations. This display is an analysis aid, not a verified incident list or dispatch alert.')
