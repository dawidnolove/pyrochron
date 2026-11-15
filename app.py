from __future__ import annotations

import sys
import json
import base64
from datetime import date
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from pyrochron.analytics import (DEFAULT_ACTIVITY_MEASURE, REGIONS, apply_activity_measure, empirical_percentile, export_hotspots, load_calendar, load_grid, load_report,
                                 make_brief_pdf, make_calendar_png, select_grid)
from pyrochron.pipeline import SENSOR_LABEL, SENSORS

st.set_page_config(page_title='Pyrochron | Fire activity, in context',page_icon=None,layout='wide',initial_sidebar_state='collapsed')
st.markdown('''
<style>
:root { color-scheme:light; --paper:#f2f4f3; --surface:#fff; --surface-raised:#f7f9f8; --ink:#24312b; --muted:#58655e; --line:#d8dfdb; --accent:#286246; --accent-soft:#e8f0eb; }
html, body, [data-testid="stAppViewContainer"] { background:var(--paper); color:var(--ink); font-size:16px; }
[data-testid="stSidebar"] { background:var(--surface-raised); border-right:1px solid var(--line); }
[data-testid="stHeader"] { background:var(--paper); }
.block-container { width:calc(100% - 2rem) !important; max-width:1180px; min-width:0; margin:1.25rem auto 2.5rem; padding:2rem 2.25rem 3rem; background:var(--surface); border:1px solid var(--line); border-radius:8px; box-shadow:0 4px 22px rgba(28,48,37,.05); }
body, p, li, label, input, button, textarea, [data-testid="stMarkdownContainer"] { font-family:Arial, "Helvetica Neue", sans-serif; }
h1,h2,h3,h4 { color:var(--ink); font-family:Arial, "Helvetica Neue", sans-serif; letter-spacing:-.025em; }
h1 { font-size:2.55rem; line-height:1.14; max-width:28ch; margin:.3rem 0 .7rem; }
h2 { font-size:1.65rem; line-height:1.25; margin-top:3.4rem; padding-top:1rem; border-top:1px solid var(--line); }
h3 { font-size:1.22rem; line-height:1.35; margin-top:1.6rem; }
p,li { color:var(--ink); line-height:1.65; }
[data-testid="stMetric"] { background:transparent; border:0; border-top:2px solid var(--accent); border-radius:0; padding:12px 12px 12px 0; }
[data-testid="stMetricLabel"] { color:var(--muted); font-size:.875rem; }
[data-testid="stMetricValue"] { color:var(--accent); font-family:Georgia,"Times New Roman",serif; }
hr { border-color:var(--line); margin:2.4rem 0; }
.stButton>button,.stDownloadButton>button { min-height:44px; border-radius:3px; border:1px solid var(--accent); color:var(--accent); background:var(--surface); font-weight:600; transition:background-color .12s ease,color .12s ease,border-color .12s ease; }
.stButton>button:hover,.stDownloadButton>button:hover { border-color:var(--accent); color:var(--surface); background:var(--accent); }
.stButton>button:active,.stDownloadButton>button:active { background:#254636; color:var(--surface); }
.stButton>button:focus-visible,.stDownloadButton>button:focus-visible,button:focus-visible,input:focus-visible,[role="combobox"]:focus-visible { outline:3px solid var(--accent); outline-offset:3px; box-shadow:none; }
button:disabled { opacity:.55; cursor:not-allowed; }
[data-testid="stTextInput"] input,[data-testid="stNumberInput"] input,[data-testid="stDateInput"] input,[data-testid="stSelectbox"] [role="combobox"] { background:var(--surface); border-color:var(--line); color:var(--ink); min-height:44px; }
[data-testid="stTextInput"] input:focus,[data-testid="stNumberInput"] input:focus,[data-testid="stDateInput"] input:focus { border-color:var(--accent); box-shadow:0 0 0 1px var(--accent); }
[data-testid="stCaptionContainer"],.small-note { font-size:.875rem; color:var(--muted); line-height:1.55; }
.eyebrow { letter-spacing:.09em; text-transform:uppercase; font-size:.75rem; color:var(--accent); font-weight:700; }
.hero-copy { max-width:66ch; font-size:1.125rem; }
.empty-state { border-left:3px solid var(--accent); background:var(--surface-raised); padding:.8rem 1rem; color:var(--ink); }
[data-testid="stSpinner"] { color:var(--accent); }
[data-testid="stDataFrame"] { border-top:1px solid var(--line); }
.pyro-nav { position:sticky; top:2.9rem; z-index:100; display:flex; gap:.45rem; overflow-x:auto; padding:.65rem .8rem; margin:0 0 2rem; border:1px solid var(--line); border-radius:5px; background:rgba(255,255,255,.97); }
.pyro-nav a { flex:0 0 auto; padding:.45rem .7rem; color:var(--accent); text-decoration:none; font-size:.86rem; font-weight:700; border-radius:3px; }
.pyro-nav a:hover,.pyro-nav a:focus-visible { background:var(--accent-soft); outline:2px solid var(--accent); outline-offset:1px; }
.section-anchor { height:1px; margin-top:1.35rem; scroll-margin-top:6.25rem; }
.site-footer { margin-top:2rem; padding-top:1rem; border-top:1px solid var(--line); color:var(--muted); font-size:.85rem; line-height:1.6; }
@media (max-width: 720px) {
  .block-container { width:100% !important; max-width:none; margin:0; padding:1.1rem 1rem 2rem; border:0; border-radius:0; box-shadow:none; }
  h1 { font-size:2rem; }
  h2 { font-size:1.55rem; margin-top:2.6rem; padding-top:.9rem; }
  h3 { font-size:1.15rem; }
  .section-anchor { margin-top:.9rem; scroll-margin-top:5.5rem; }
  [data-testid="stMetric"] { padding-top:.65rem; }
  .stButton>button,.stDownloadButton>button { width:100%; }
  .pyro-nav { top:2.7rem; margin-left:-.5rem; margin-right:-.5rem; }
}
@media (prefers-reduced-motion: reduce) { *,*::before,*::after { scroll-behavior:auto !important; transition:none !important; animation:none !important; } }
@page { size:A4 portrait; margin:14mm; }
@media print {
  html,body,[data-testid="stAppViewContainer"],.main { background:#fff !important; color:#111 !important; font-size:10pt !important; }
  .block-container { width:100% !important; max-width:none !important; min-width:0 !important; margin:0 !important; padding:0 !important; }
  [data-testid="stSidebar"],[data-testid="stHeader"],.pyro-nav,
  [data-testid="stDownloadButton"],[data-testid="stButton"],[data-testid="stFormSubmitButton"],
  [data-testid="stIFrame"],iframe { display:none !important; }
  h1 { font-size:24pt !important; max-width:none !important; }
  h2 { font-size:17pt !important; margin-top:18pt !important; padding-top:0 !important; border-top:0 !important; break-after:avoid; }
  .section-anchor { height:0 !important; margin-top:8pt !important; }
  h3 { font-size:13pt !important; margin-top:12pt !important; break-after:avoid; }
  [data-testid="stMetric"] { break-inside:avoid; page-break-inside:avoid; }
  [data-testid="stHorizontalBlock"] { display:block !important; }
  [data-testid="stColumn"] { width:100% !important; max-width:100% !important; flex:1 1 100% !important; }
  [data-testid="stPlotlyChart"],[data-testid="stVegaLiteChart"],[data-testid="stAltairChart"],[data-testid="stImage"],
  [data-testid="stDataFrame"],[data-testid="stTable"] { max-width:100% !important; break-inside:avoid; page-break-inside:avoid; }
  [data-testid="stExpander"] details > div { display:block !important; }
  img { max-width:100% !important; height:auto !important; max-height:230mm !important; object-fit:contain !important; }
  a { color:#111 !important; text-decoration:none !important; }
  p,li { font-size:9.5pt !important; line-height:1.4 !important; }
  hr { margin:12pt 0 !important; }
}
</style>
''',unsafe_allow_html=True)

@st.cache_data(show_spinner='Loading the precomputed fire record…')
def data_bundle():
    return load_grid(),load_calendar(),load_report()

grid,calendar,report=data_bundle()
last_rows=[]
for item in report.get('source_audit',[]):
    if item.get('date_max'): last_rows.append(pd.Timestamp(item['date_max']))
latest_data=max(last_rows) if last_rows else pd.Timestamp('2024-12-31')
latest_complete=(latest_data.to_period('M').start_time-pd.offsets.MonthBegin(1)) if latest_data.day<latest_data.days_in_month else latest_data.to_period('M').start_time

min_date=date(2000,11,1)
max_date=latest_data.date()
end_default=min(max_date,date(2026,10,31))
recent_start=max(min_date,(pd.Timestamp(max_date)-pd.DateOffset(years=5)).date())
if 'applied_region' not in st.session_state:
    st.session_state.applied_region=list(REGIONS)[0]
    st.session_state.applied_start=recent_start
    st.session_state.applied_end=end_default
    st.session_state.applied_custom=False
    st.session_state.applied_bbox=tuple(REGIONS[st.session_state.applied_region]['bbox'])
    st.session_state.applied_profile='Recent activity'
    st.session_state.applied_period='Most recent 5 years'
    st.session_state.applied_season='All months'
    st.session_state.applied_months=list(range(1,13))
    st.session_state.applied_daypart='All times'
    st.session_state.applied_sensor_focus='All sensors'
    st.session_state.applied_measure='Approximate daily 1 km clusters'
st.session_state.setdefault('applied_profile','Recent activity')
if st.session_state.applied_profile=='Fire response':
    st.session_state.applied_profile='Recent activity'
st.session_state.setdefault('applied_period','Most recent 5 years')
st.session_state.setdefault('applied_season','All months')
st.session_state.setdefault('applied_months',list(range(1,13)))
st.session_state.setdefault('applied_daypart','All times')
st.session_state.setdefault('applied_sensor_focus','All sensors')
st.session_state.setdefault('applied_measure','Approximate daily 1 km clusters')
if st.session_state.get('measure_default_version',0)<1:
    st.session_state.applied_measure='Approximate daily 1 km clusters'
    st.session_state.measure_default_version=1

st.markdown('<div class="eyebrow">NASA FIRMS / MODIS and VIIRS / Four configured study areas</div>',unsafe_allow_html=True)
st.title('Compare monthly fire records across changing satellites.')
st.markdown('<div class="hero-copy">Pyrochron puts active-fire detections on a shared monthly grid, compares sensor counts against MODIS, and keeps each source visible.</div>',unsafe_allow_html=True)
operating_mode=st.radio('Analysis context',['Historical benchmark regions','Live custom ROI · last 7 days'],
    horizontal=True,key='operating_mode',help='Historical mode reads local precomputed archives (2000–2026). Live ROI queries NASA FIRMS NRT sources.')
if operating_mode=='Live custom ROI · last 7 days':
    from pyrochron.roi_view import render_live_roi
    render_live_roi()
    st.stop()
st.caption('Historical benchmark mode · four configured regions · local precomputed observations from 2000–2026. This is historical context, not a live alert.')
st.caption('Default view: spatially collapsed VIIRS repeat-detection proxy, the better-scoring candidate on the recorded 2020–2024 holdout. Count-aligned HFAI remains selectable for comparison. Both measure satellite detections, not confirmed fire incidents.')

with st.sidebar:
    st.markdown('## Explore the record')
    # A regular sidebar container keeps the date widgets reactive. Streamlit forms
    # buffer selectbox changes until submit, which prevented the custom date fields
    # from appearing when the user chose "Custom range".
    with st.container():
        profile_options=['Recent activity','Seasonal research','Land management']
        profile_choice=st.selectbox('View profile',profile_options,index=profile_options.index(st.session_state.applied_profile))
        st.caption({'Recent activity':'Recent five-year archive context; this is not live dispatch data.', 'Seasonal research':'Full-record default.', 'Land management':'Full record with harvest context for Punjab.'}[profile_choice])
        region_choice=st.selectbox('Study area',list(REGIONS),index=list(REGIONS).index(st.session_state.applied_region),format_func=lambda x:REGIONS[x]['label'])
        full_bbox=REGIONS[region_choice]['bbox']
        custom_choice=st.toggle('Set a smaller box inside this archive area',value=st.session_state.applied_custom)
        if custom_choice:
            with st.expander('Bounding box · west, south, east, north',expanded=True):
                current_bbox=st.session_state.applied_bbox if st.session_state.applied_region==region_choice else full_bbox
                west_choice=st.number_input('West longitude',min_value=-180.0,max_value=180.0,value=float(current_bbox[0]),step=.25,format='%.2f')
                south_choice=st.number_input('South latitude',min_value=-90.0,max_value=90.0,value=float(current_bbox[1]),step=.25,format='%.2f')
                east_choice=st.number_input('East longitude',min_value=-180.0,max_value=180.0,value=float(current_bbox[2]),step=.25,format='%.2f')
                north_choice=st.number_input('North latitude',min_value=-90.0,max_value=90.0,value=float(current_bbox[3]),step=.25,format='%.2f')
        period_options=['Custom range','Most recent 5 years','Full record','Calibration + final test (2012–2024)']
        period_choice=st.selectbox('Date range preset',period_options,index=period_options.index(st.session_state.applied_period))
        if period_choice=='Custom range':
            start_choice=st.date_input('From date',value=st.session_state.applied_start,min_value=min_date,max_value=max_date,key='start')
            end_choice=st.date_input('Through date',value=st.session_state.applied_end,min_value=min_date,max_value=max_date,key='end')
            st.caption('The data are monthly; choosing any day includes that entire calendar month.')
        season_options=['All months','Boreal summer (Jun–Aug)','Southern dry season (Sep–Mar)','Punjab harvest context (Oct–Nov)','Choose months']
        season_choice=st.selectbox('Season or event window',season_options,index=season_options.index(st.session_state.applied_season))
        month_names={i:pd.Timestamp(2000,i,1).strftime('%B') for i in range(1,13)}
        if season_choice=='Choose months':
            chosen_months=st.multiselect('Months to include',[month_names[i] for i in range(1,13)],default=[month_names[i] for i in st.session_state.applied_months])
        daypart_choice=st.selectbox('Time of day for source detail',['All times','Day detections','Night detections'],index=['All times','Day detections','Night detections'].index(st.session_state.applied_daypart))
        sensor_choice=st.selectbox('Satellite data to emphasize',['All sensors','MODIS','VIIRS'],index=['All sensors','MODIS','VIIRS'].index(st.session_state.applied_sensor_focus))
        measure_options=['Approximate daily 1 km clusters','Count-aligned HFAI']
        measure_choice=st.selectbox('Activity measure',measure_options,index=measure_options.index(st.session_state.applied_measure),help='Default: cluster collapse, the better-scoring of these two measures on the 2020–2024 holdout (MAE 4.89; R² 0.920). It groups VIIRS detections in approximate 1 km cells by date and day/night; it does not count unique fires.')
        st.caption('MODIS/VIIRS is the available sensor type. This dataset does not classify wildfire, crop burning, or ignition cause.')
        update_filters=st.button('Apply filters',type='primary',use_container_width=True)

filter_feedback=[]
if update_filters:
    profile_changed=profile_choice!=st.session_state.applied_profile
    if profile_changed and profile_choice=='Recent activity':
        period_choice='Most recent 5 years'
        season_choice='All months'
    elif profile_changed and profile_choice=='Seasonal research':
        period_choice='Full record'
        season_choice='All months'
    elif profile_changed and profile_choice=='Land management':
        period_choice='Full record'
        season_choice='Punjab harvest context (Oct–Nov)' if region_choice=='Punjab' else 'All months'
    if period_choice=='Most recent 5 years':
        start_choice=recent_start
        end_choice=end_default
    elif period_choice=='Full record':
        start_choice=min_date
        end_choice=max_date
    elif period_choice=='Calibration + final test (2012–2024)':
        start_choice=date(2012,1,1)
        end_choice=date(2024,12,31)
    if start_choice>end_choice:
        start_choice,end_choice=end_choice,start_choice
        filter_feedback.append('The date range was reversed to keep the start before the end.')
    if season_choice=='All months':
        active_months=list(range(1,13))
    elif season_choice=='Boreal summer (Jun–Aug)':
        active_months=[6,7,8]
    elif season_choice=='Southern dry season (Sep–Mar)':
        active_months=[9,10,11,12,1,2,3]
    elif season_choice=='Punjab harvest context (Oct–Nov)':
        active_months=[10,11]
    else:
        active_months=[i for i in range(1,13) if month_names[i] in chosen_months]
        if not active_months:
            active_months=list(range(1,13))
            filter_feedback.append('Choose at least one month; all months were selected.')
    if custom_choice:
        entered_bbox=(float(west_choice),float(south_choice),float(east_choice),float(north_choice))
        if entered_bbox[0]>=entered_bbox[2] or entered_bbox[1]>=entered_bbox[3]:
            entered_bbox=tuple(full_bbox)
            filter_feedback.append('Invalid bounding box; the full study area was selected.')
        else:
            clipped=(max(entered_bbox[0],full_bbox[0]),max(entered_bbox[1],full_bbox[1]),min(entered_bbox[2],full_bbox[2]),min(entered_bbox[3],full_bbox[3]))
            if clipped[0]>=clipped[2] or clipped[1]>=clipped[3]:
                entered_bbox=tuple(full_bbox)
                filter_feedback.append('The box was outside the source coverage; the full study area was selected.')
            else:
                entered_bbox=clipped
    else:
        entered_bbox=tuple(full_bbox)
    st.session_state.applied_region=region_choice
    st.session_state.applied_start=start_choice
    st.session_state.applied_end=end_choice
    st.session_state.applied_custom=custom_choice
    st.session_state.applied_bbox=entered_bbox
    st.session_state.applied_profile=profile_choice
    st.session_state.applied_period=period_choice
    st.session_state.applied_season=season_choice
    st.session_state.applied_months=active_months
    st.session_state.applied_daypart=daypart_choice
    st.session_state.applied_sensor_focus=sensor_choice
    st.session_state.applied_measure=measure_choice
    st.session_state.pop('point_export',None)

region=st.session_state.applied_region
bbox=tuple(st.session_state.applied_bbox)
start_date=st.session_state.applied_start
end_date=st.session_state.applied_end
with st.sidebar:
    st.divider()
    st.caption(f'{st.session_state.applied_profile} · {REGIONS[region]["label"]} · {start_date:%b %Y} – {end_date:%b %Y}')
    for message in filter_feedback:
        st.warning(message)
    st.caption(f'Season: {st.session_state.applied_season} · {st.session_state.applied_daypart.lower()} · {st.session_state.applied_sensor_focus.lower()} · {st.session_state.applied_measure.lower()}')
    profile_notes={'Recent activity':'Starts with a recent five-year archive view; it is not a live incident feed.',
                   'Seasonal research':'Starts with the full record for long-term seasonal comparison.',
                   'Land management':'Starts with a full record and Punjab harvest-month context when Punjab is selected.'}
    st.caption(profile_notes[st.session_state.applied_profile])
    st.caption('Use Apply filters to update your view. Month windows are seasonal context; satellite detections are not classified by fire cause.')
start_iso=pd.Timestamp(start_date).to_period('M').start_time.date().isoformat()
end_iso=pd.Timestamp(end_date).to_period('M').end_time.date().isoformat()
st.caption(f'Current view: {REGIONS[region]["label"]} · {start_date:%b %Y}–{end_date:%b %Y} · {st.session_state.applied_season} · {st.session_state.applied_measure}. Open the filter panel from the upper-left control to change it. For a clean printout, use your browser’s Print / Save as PDF command.')

active_measure=DEFAULT_ACTIVITY_MEASURE if st.session_state.applied_measure=='Approximate daily 1 km clusters' else 'count_aligned_hfai'
grid=apply_activity_measure(grid,report,active_measure)

selected=select_grid(grid,region,bbox,start_iso,end_iso)
selected=selected[selected.month.isin(st.session_state.applied_months)].copy()
selected_cal=(selected.groupby(['region','year','month'],as_index=False)
             .agg(hfai=('hfai','sum'),count_aligned_hfai=('count_aligned_hfai','sum'),cluster_hfai=('cluster_hfai','sum'),raw_sensor_detections=('raw_sensor_detections','sum'),
                  MODIS_count=('MODIS_count','sum'),SNPP_VIIRS_count=('SNPP_VIIRS_count','sum'),
                  NOAA20_VIIRS_count=('NOAA20_VIIRS_count','sum'),NOAA21_VIIRS_count=('NOAA21_VIIRS_count','sum'),
                  MODIS_day_count=('MODIS_day_count','sum'),MODIS_night_count=('MODIS_night_count','sum'),
                  MODIS_Terra_count=('MODIS_Terra_count','sum'),MODIS_Aqua_count=('MODIS_Aqua_count','sum'),
                  MODIS_Terra_day_count=('MODIS_Terra_day_count','sum'),MODIS_Terra_night_count=('MODIS_Terra_night_count','sum'),
                  MODIS_Aqua_day_count=('MODIS_Aqua_day_count','sum'),MODIS_Aqua_night_count=('MODIS_Aqua_night_count','sum'),
                  SNPP_VIIRS_day_count=('SNPP_VIIRS_day_count','sum'),SNPP_VIIRS_night_count=('SNPP_VIIRS_night_count','sum'),
                  NOAA20_VIIRS_day_count=('NOAA20_VIIRS_day_count','sum'),NOAA20_VIIRS_night_count=('NOAA20_VIIRS_night_count','sum'),
                  NOAA21_VIIRS_day_count=('NOAA21_VIIRS_day_count','sum'),NOAA21_VIIRS_night_count=('NOAA21_VIIRS_night_count','sum'),
                  active_sensor_count=('active_sensor_count','max')))
selected_cal['month_start']=pd.to_datetime(dict(year=selected_cal.year,month=selected_cal.month,day=1))
ref_source=grid[(grid.region.eq(region)) & (grid.year.between(2012,2019)) &
                grid.cell_lon_i.between(np.floor(bbox[0]*4),np.ceil(bbox[2]*4)-1) &
                grid.cell_lat_i.between(np.floor(bbox[1]*4),np.ceil(bbox[3]*4)-1) &
                grid.month.isin(st.session_state.applied_months)]
ref_calendar=ref_source.groupby(['year','month'],as_index=False).agg(hfai=('hfai','sum'))
ref_by_month=ref_calendar.groupby('month').hfai.median()
comparison_reference=ref_source.groupby(['year','month'],as_index=False).agg(count_aligned_hfai=('count_aligned_hfai','sum'),cluster_hfai=('cluster_hfai','sum'))
comparison_baselines=comparison_reference.groupby('month')[['count_aligned_hfai','cluster_hfai']].median()
selected_months=selected_cal.copy()
selected_months['baseline']=selected_months.month.map(ref_by_month)
selected_total=float(selected_months.hfai.sum()) if not selected_months.empty else 0.0
expected_total=float(selected_months.baseline.fillna(0).sum()) if not selected_months.empty else 0.0
period_ratio=selected_total/expected_total if expected_total>0 else np.nan

st.markdown('''<nav class="pyro-nav" aria-label="Jump to a project section">
<a href="#monthly-activity">Activity</a><a href="#study-areas">Study areas</a><a href="#landscape-gallery">Landscapes</a><a href="#method-validation">Method & validation</a><a href="#downloads">Downloads</a><a href="#api-access">API</a>
</nav>''',unsafe_allow_html=True)

hero_main,hero_context=st.columns([2,1],gap='large')
measure_label='HFAI' if st.session_state.applied_measure=='Count-aligned HFAI' else 'Daily cluster proxy'
with hero_main:
    metric_value,metric_reference=st.columns(2,gap='medium')
    metric_value.metric(f'{measure_label} in this period',f'{selected_total:,.0f}',help='Sum across the selected 0.25-degree cells and months. Cluster proxy groups same-sensor VIIRS detections in approximate 1 km daily cells; neither measure counts unique fires.')
    metric_reference.metric(f'{measure_label} vs same-month baseline',f'{period_ratio:.2f}x' if np.isfinite(period_ratio) else 'Unavailable',help='Expected activity sums the fixed 2012-2019 median for each selected calendar month using the same selected measure.')
with hero_context.container(border=True):
    st.markdown('<div class="eyebrow">Selected record</div>',unsafe_allow_html=True)
    st.markdown(f'**{REGIONS[region]["label"]}**  \n{start_date:%B %Y} to {end_date:%B %Y}')
    st.caption(f'{int(selected.raw_sensor_detections.sum()):,} quality-filtered FIRMS detections in {selected[["cell_lat_i","cell_lon_i"]].drop_duplicates().shape[0]:,} active grid cells. The selected index estimates comparable detection activity; it does not identify unique fires.')

if not selected_cal.empty:
    with st.expander('Compare count alignment with the repeat-detection proxy'):
        comparison_reference_for_selection=comparison_baselines.reindex(selected_cal.month.to_numpy())
        count_total=float(selected_cal.count_aligned_hfai.sum())
        cluster_total=float(selected_cal.cluster_hfai.sum())
        count_normal=float(comparison_reference_for_selection.count_aligned_hfai.sum())
        cluster_normal=float(comparison_reference_for_selection.cluster_hfai.sum())
        compare_count,compare_cluster=st.columns(2)
        compare_count.metric('Count-aligned HFAI',f'{count_total:,.1f}',f'{count_total/count_normal:.2f}× its seasonal reference' if count_normal>0 else 'reference unavailable')
        compare_cluster.metric('Approximate daily 1 km clusters',f'{cluster_total:,.1f}',f'{cluster_total/cluster_normal:.2f}× its seasonal reference' if cluster_normal>0 else 'reference unavailable')
        compare_long=selected_cal[['month_start','count_aligned_hfai','cluster_hfai']].melt(id_vars='month_start',var_name='measure',value_name='activity')
        compare_long['measure']=compare_long.measure.map({'count_aligned_hfai':'Count-aligned HFAI','cluster_hfai':'Daily cluster proxy'})
        compare_chart=(alt.Chart(compare_long).mark_line(point=True).encode(x=alt.X('month_start:T',title=None),y=alt.Y('activity:Q',title='Monthly activity estimate'),color=alt.Color('measure:N',title='Measure'),tooltip=['month_start:T','measure:N',alt.Tooltip('activity:Q',format=',.1f')]).properties(height=220))
        st.altair_chart(compare_chart,width='stretch')
        st.caption('Each total is compared with its own 2012–2019 same-month reference. The cluster view collapses some same-pass repeats; neither line represents confirmed fire incidents or burned area.')

st.markdown('<div id="monthly-activity" class="section-anchor"></div>',unsafe_allow_html=True)
if selected.empty:
    st.markdown('<div class="empty-state"><strong>No detections in this selection.</strong><br>Try widening the date range or returning to the full archive rectangle.</div>',unsafe_allow_html=True)
else:
    st.markdown(f'## Monthly {measure_label.lower()} in the selected area')
    if st.session_state.applied_measure=='Approximate daily 1 km clusters':
        st.markdown('VIIRS detections are collapsed to one record per approximate 1 km location, acquisition date and day/night period, then calibrated against MODIS over the training overlap. This reduces repeat detections in a pass; it cannot merge a fire across time or establish incident identity. The comparison baseline uses the same cluster measure.')
    else:
        st.markdown('Each square sums calibrated satellite detection activity across the selected area for one calendar month. The historical reference is fixed at 2012–2019, before the separate 2020–2024 sensor evaluation.')
    calendar_view=st.radio('Calendar view', ['Year × month','Annual totals','Seasonal profile'], horizontal=True, key='calendar_view')
    if calendar_view=='Year × month':
        heat=selected_cal.copy(); heat['month_name']=heat.month.map(lambda x:pd.Timestamp(2000,x,1).strftime('%b'))
        chart=(alt.Chart(heat).mark_rect(cornerRadius=1,stroke='#f5f4ef',strokeWidth=1)
            .encode(x=alt.X('month_name:O',sort=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'],title=None),
                    y=alt.Y('year:O',sort='descending',title=None),
                    color=alt.Color('hfai:Q',title=f'{measure_label} activity',scale=alt.Scale(domainMin=0,range=['#829985','#718b77','#5c7b64','#315a49']),legend=alt.Legend(orient='bottom')),
                    tooltip=[alt.Tooltip('year:O',title='Year'),alt.Tooltip('month_name:O',title='Month'),alt.Tooltip('hfai:Q',title=measure_label,format=',.1f'),alt.Tooltip('raw_sensor_detections:Q',title='Raw detections',format=',')])
            .properties(height=max(260,min(570,len(heat.year.unique())*22)),width='container')
            .configure_view(stroke=None).configure_axis(domain=False,tickSize=0,grid=False,labelColor='#58625c',titleColor='#58625c'))
        st.altair_chart(chart,width='stretch')
    elif calendar_view=='Annual totals':
        annual=selected_cal.groupby('year',as_index=False).agg(activity=('hfai','sum'))
        annual_chart=alt.Chart(annual).mark_bar(color='#315a49').encode(
            x=alt.X('year:O',title='Year'),y=alt.Y('activity:Q',title=f'Selected-month {measure_label} total'),
            tooltip=[alt.Tooltip('year:O',title='Year'),alt.Tooltip('activity:Q',title='Activity',format=',.1f')]
        ).properties(height=280)
        st.altair_chart(annual_chart,width='stretch')
        st.caption('Each bar sums only the months and years in the applied filters; incomplete edge years may contain fewer months.')
    else:
        active_months=sorted(selected_cal.month.unique())
        seasonal=pd.concat([
            selected_cal.groupby('month',as_index=False).agg(activity=('hfai','median')).assign(period='Selected period'),
            ref_calendar[ref_calendar.month.isin(active_months)].groupby('month',as_index=False).agg(activity=('hfai','median')).assign(period='2012–2019 median')
        ],ignore_index=True)
        seasonal['month_name']=seasonal.month.map(lambda x:pd.Timestamp(2000,int(x),1).strftime('%b'))
        seasonal_chart=alt.Chart(seasonal).mark_line(point=True).encode(
            x=alt.X('month:O',sort=active_months,title='Calendar month',axis=alt.Axis(labelExpr="['','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][datum.value]")),
            y=alt.Y('activity:Q',title=f'Median monthly {measure_label}'),color=alt.Color('period:N',title='Period',scale=alt.Scale(domain=['Selected period','2012–2019 median'],range=['#315a49','#9a7450'])),
            tooltip=[alt.Tooltip('month_name:N',title='Month'),alt.Tooltip('period:N',title='Period'),alt.Tooltip('activity:Q',title='Activity',format=',.1f')]
        ).properties(height=280)
        st.altair_chart(seasonal_chart,width='stretch')
        st.caption('The reference is the 2012–2019 same-month median for the same area, spatial extent, activity measure and selected months.')
    current_peak=int(selected_cal.groupby('month').hfai.mean().idxmax()) if not selected_cal.empty else None
    baseline_peak=int(ref_by_month.idxmax()) if not ref_by_month.empty else None
    if current_peak and baseline_peak:
        st.caption(f'Season timing: the highest mean activity in this selected span is {pd.Timestamp(2000,current_peak,1):%B}; the fixed 2012–2019 median peaks in {pd.Timestamp(2000,baseline_peak,1):%B}.')
    complete=selected_cal[selected_cal.month_start.eq(latest_complete)]
    if not complete.empty:
        baseline_value=float(ref_by_month.get(int(latest_complete.month),np.nan))
        latest_hfai=float(complete.hfai.sum())
        historical_same_month=ref_calendar.loc[ref_calendar.month.eq(int(latest_complete.month)),'hfai']
        percentile=empirical_percentile(latest_hfai,historical_same_month)
        st.metric(f'Latest complete month · {latest_complete:%B %Y}',f'{latest_hfai:,.0f}',
                  f'{latest_hfai/baseline_value:.2f}× same-month 2012–2019 median' if baseline_value>0 else 'same-month median unavailable')
        if percentile is not None:
            st.caption(f'This month is at the {percentile:.0f}th empirical percentile among {len(historical_same_month)} same-calendar-month HFAI values from 2012–2019. The percentile is descriptive and uses a short reference sample; it is not a forecast.')

    st.markdown('### Source and day/night comparison')
    source_columns={'MODIS':{'All times':'MODIS_count','Day detections':'MODIS_day_count','Night detections':'MODIS_night_count'},
                    'MODIS_Terra':{'All times':'MODIS_Terra_count','Day detections':'MODIS_Terra_day_count','Night detections':'MODIS_Terra_night_count'},
                    'MODIS_Aqua':{'All times':'MODIS_Aqua_count','Day detections':'MODIS_Aqua_day_count','Night detections':'MODIS_Aqua_night_count'},
                    'SNPP_VIIRS':{'All times':'SNPP_VIIRS_count','Day detections':'SNPP_VIIRS_day_count','Night detections':'SNPP_VIIRS_night_count'},
                    'NOAA20_VIIRS':{'All times':'NOAA20_VIIRS_count','Day detections':'NOAA20_VIIRS_day_count','Night detections':'NOAA20_VIIRS_night_count'},
                    'NOAA21_VIIRS':{'All times':'NOAA21_VIIRS_count','Day detections':'NOAA21_VIIRS_day_count','Night detections':'NOAA21_VIIRS_night_count'}}
    visible_sources=[name for name in source_columns if st.session_state.applied_sensor_focus=='All sensors' or (name.startswith('MODIS') if st.session_state.applied_sensor_focus=='MODIS' else not name.startswith('MODIS'))]
    source_map={name:source_columns[name][st.session_state.applied_daypart] for name in visible_sources}
    source_long=selected_cal.melt(id_vars=['year','month'],value_vars=list(source_map.values()),var_name='column',value_name='detections')
    reverse_sources={column:name for name,column in source_map.items()}
    source_long['source']=source_long.column.map(reverse_sources)
    source_long['source']=source_long['source'].str.replace('_count','',regex=False).str.replace('_day','',regex=False).str.replace('_night','',regex=False)
    source_title='Filtered detections' if st.session_state.applied_daypart=='All times' else st.session_state.applied_daypart
    source_chart=alt.Chart(source_long).mark_line(point=True,strokeWidth=2).encode(x=alt.X('year:O',title='Year'),y=alt.Y('detections:Q',title=source_title),color=alt.Color('source:N',title='Source',scale=alt.Scale(domain=['MODIS','MODIS_Terra','MODIS_Aqua','SNPP_VIIRS','NOAA20_VIIRS','NOAA21_VIIRS'],range=['#315a49','#424b46','#59615b','#6b736d','#7d857f','#8c938e'])),tooltip=['year:O','source:N','detections:Q']).properties(height=210)
    st.altair_chart(source_chart,width='stretch')
    daynight=[]
    daynight_sources=['MODIS','SNPP_VIIRS','NOAA20_VIIRS','NOAA21_VIIRS']
    if st.session_state.applied_sensor_focus=='MODIS': daynight_sources=['MODIS']
    elif st.session_state.applied_sensor_focus=='VIIRS': daynight_sources=['SNPP_VIIRS','NOAA20_VIIRS','NOAA21_VIIRS']
    periods=['Day','Night'] if st.session_state.applied_daypart=='All times' else [st.session_state.applied_daypart.split()[0]]
    for sensor in daynight_sources:
        for period in periods:
            daynight.append({'source':sensor,'period':period,'count':int(selected[f'{sensor}_{period.lower()}_count'].sum())})
    dn=pd.DataFrame(daynight)
    st.altair_chart(alt.Chart(dn).mark_bar().encode(x=alt.X('source:N',title='Source'),y=alt.Y('count:Q',title='Filtered detections'),color=alt.Color('period:N',scale=alt.Scale(domain=['Day','Night'],range=['#315a49','#7d857f'])),xOffset='period:N',tooltip=['source:N','period:N','count:Q']).properties(height=190),width='stretch')
    st.caption('Source charts follow your sensor and day/night filters. The selected activity measure uses all active sources and the full monthly record; day/night filters do not alter its calibration.')

    rain_chart_data=None
    left,right=st.columns([1.05,.95],gap='large')
    with left:
        st.markdown('### Where the record is concentrated')
        map_data=(selected.groupby(['lat','lon'],as_index=False).agg(hfai=('hfai','sum'),raw=('raw_sensor_detections','sum')))
        map_data=map_data[map_data.hfai>0]
        if not map_data.empty:
            map_data=map_data.rename(columns={'lat':'latitude','lon':'longitude','hfai':'weight'})
            st.map(map_data,latitude='latitude',longitude='longitude',size='weight',color='#315a49',zoom=3)
            st.caption('Symbols mark 0.25° cell centres. They show monthly active-fire detections accumulated over the selected dates, not fire perimeters.')
        else:
            st.info('No active-fire detections are mapped for this selection.')
    with right:
        st.markdown('### Weather beside the fire record')
        weather_cols={'power_precip_anomaly_pct':'rainfall_departure','power_temp_anomaly_c':'temperature_departure'}
        if all(c in selected.columns for c in weather_cols):
            weather=(selected.drop_duplicates(['year','month'])[['year','month','power_precip_mm_month','power_precip_anomaly_pct','power_precip_anomaly_mm','power_temp_anomaly_c']]
                     .sort_values(['year','month']))
            if not weather.empty and weather.power_precip_mm_month.notna().any():
                latest=weather.iloc[-1]
                a,b=st.columns(2)
                a.metric('Rainfall, selected period',f'{weather.power_precip_mm_month.mean():.0f} mm/month',help='NASA POWER monthly mean daily precipitation × days in month, averaged across selected months.')
                b.metric('Temperature departure',f'{weather.power_temp_anomaly_c.mean():+.1f} °C' if pd.notna(weather.power_temp_anomaly_c.mean()) else '—',help='Monthly mean temperature relative to the 2001–2020 same-month median at the area centroid.')
                if pd.notna(latest.power_precip_anomaly_pct):
                    st.caption(f'For the latest month in view, precipitation was {latest.power_precip_anomaly_pct:+.0f}% from its 2001–2020 same-month median.')
                elif pd.notna(latest.get('power_precip_anomaly_mm')):
                    st.caption(f'For the latest month in view, precipitation was {latest.power_precip_anomaly_mm:+.1f} mm from its 2001–2020 same-month median. Percentage departures are suppressed when the normal is very small.')
                else:
                    st.caption('Weather context is unavailable for this period.')
                rain_chart_data=weather.tail(96).copy()
                rain_chart_data['month_start']=pd.to_datetime(dict(year=rain_chart_data.year,month=rain_chart_data.month,day=1))
                st.caption('NASA POWER MERRA-2 monthly values at the area centroid; context only, not a causal explanation or forecast.')
            else:
                st.info('Weather context is not available for the selected period.')
        else:
            st.info('NASA POWER weather context was not loaded.')

    if rain_chart_data is not None:
        st.markdown('### Monthly precipitation departure')
        st.caption('Difference from the 2001–2020 median for the same calendar month. Hover over a point for its month and value.')
        rain_line=alt.Chart(rain_chart_data).mark_line(color='#315a49',point=alt.OverlayMarkDef(size=28)).encode(
            x=alt.X('month_start:T',title=None,axis=alt.Axis(format='%Y',tickCount=10,labelAngle=0)),
            y=alt.Y('power_precip_anomaly_mm:Q',title='mm vs. normal',axis=alt.Axis(titleFontSize=11,labelFontSize=10)),
            tooltip=[alt.Tooltip('month_start:T',title='Month'),alt.Tooltip('power_precip_anomaly_mm:Q',title='Departure (mm)',format='+.1f')]
        ).properties(height=250)
        rain_zero=alt.Chart(pd.DataFrame({'baseline':[0]})).mark_rule(color='#8c948e',strokeDash=[4,3]).encode(y='baseline:Q')
        st.altair_chart(rain_line+rain_zero,width='stretch')

if region=='Punjab':
    st.markdown('### Punjab harvest-season context')
    seasonal=selected_cal[selected_cal.month.isin([10,11])].copy()
    if not seasonal.empty:
        seasonal['month_label']=seasonal.month.map({10:'October',11:'November'})
        st.altair_chart(alt.Chart(seasonal).mark_line(point=True,color='#315a49').encode(x=alt.X('year:O',title='Year'),y=alt.Y('hfai:Q',title='HFAI detections'),strokeDash=alt.StrokeDash('month_label:N',title='Month',scale=alt.Scale(domain=['October','November'],range=[[5,3],[1,2]])),tooltip=['year:O','month_label:N','hfai:Q','raw_sensor_detections:Q']).properties(height=220),width='stretch')
        st.caption('This seasonal pattern provides harvest-season context; the hotspot records do not identify individual crop-residue fires.')
    else:
        st.info('The applied filters contain no October or November data. Include those months to view the harvest-season context.')

st.markdown('<div id="study-areas" class="section-anchor"></div>',unsafe_allow_html=True)
st.markdown('## Four study areas, one comparison rule')
st.markdown('Case values sum each selected month\'s regional median from 2012-2019. They describe the index against its own short reference, not area burned.')
case_specs=[('Quebec','May-August 2023',[5,6,7,8],2023),
            ('Valparaiso','February 2024',[2],2024),
            ('Pantanal','September-November 2024',[9,10,11],2024),
            ('Punjab','October-November 2024',[10,11],2024)]
case_values=[]
for case_region,period_label,months,year in case_specs:
    frame=grid[(grid.region.eq(case_region)) & grid.month.isin(months)]
    target=frame[frame.year.eq(year)].groupby('month').hfai.sum()
    reference=(frame[frame.year.between(2012,2019)].groupby(['year','month']).hfai.sum()
               .groupby('month').median())
    value=float(target.sum())
    base=float(reference.reindex(months).fillna(0).sum())
    case_values.append({'region':case_region,'period':period_label,'months':months,'year':year,
                        'hfai':value,'baseline':base,'ratio':value/base if base else np.nan})
lead=case_values[0]
lead_copy,lead_figure=st.columns([.9,1.1],gap='large')
with lead_copy:
    st.markdown('<div class="eyebrow">Featured case / boreal Canada</div>',unsafe_allow_html=True)
    st.markdown('### Quebec, May-August 2023')
    st.metric(measure_label,f'{lead["hfai"]:,.0f}',f'{lead["ratio"]:.2f}x the same-month reference')
    st.markdown(f'The reference sums to {lead["baseline"]:,.2f} across the four months. The ratio is large partly because this historical baseline is small; it is not a burned-area multiplier or a typical anomaly size.')
with lead_figure:
    lead_frame=grid[(grid.region.eq('Quebec'))&grid.month.isin([5,6,7,8])]
    lead_target=lead_frame[lead_frame.year.eq(2023)].groupby('month',as_index=False).agg(activity=('hfai','sum')).assign(series='May–August 2023')
    lead_reference=(lead_frame[lead_frame.year.between(2012,2019)].groupby(['year','month'],as_index=False).hfai.sum()
                    .groupby('month',as_index=False).agg(activity=('hfai','median')).assign(series='2012–2019 same-month median'))
    lead_chart_data=pd.concat([lead_target,lead_reference],ignore_index=True)
    lead_chart_data['month_name']=lead_chart_data.month.map(lambda m:pd.Timestamp(2000,int(m),1).strftime('%b'))
    case_chart=alt.Chart(lead_chart_data).mark_bar().encode(
        x=alt.X('month_name:O',sort=['May','Jun','Jul','Aug'],title=None),
        xOffset=alt.XOffset('series:N'),y=alt.Y('activity:Q',title=measure_label),
        color=alt.Color('series:N',title='Period',scale=alt.Scale(domain=['May–August 2023','2012–2019 same-month median'],range=['#315a49','#9a7450'])),
        tooltip=['month_name:N','series:N',alt.Tooltip('activity:Q',format=',.1f')]
    ).properties(height=260)
    st.altair_chart(case_chart,width='stretch')
other_cases=pd.DataFrame([{k:v for k,v in c.items() if k in ('region','period','hfai','baseline','ratio')} for c in case_values[1:]])
other_cases=other_cases.rename(columns={'region':'Study area','period':'Period','hfai':measure_label,'baseline':'Reference sum','ratio':'Multiple'})
st.dataframe(other_cases.style.format({measure_label:'{:,.0f}','Reference sum':'{:,.2f}','Multiple':'{:.2f}x'}),hide_index=True,width='stretch')
st.caption('Valparaiso, the Pantanal and Punjab remain available as comparison cases. Punjab is harvest-season context; hotspots are not classified as crop-residue fires. No case identifies ignition cause.')

st.markdown('<div id="landscape-gallery" class="section-anchor"></div>',unsafe_allow_html=True)
st.markdown('## Landscapes across the four study areas')
st.caption('Select a card to turn it over and read a short description. Selecting another card returns the others to their photographs.')
gallery_specs=[
    ('Brazil','Pantanal','brazil.jpg','Vast tropical wetland filled with rivers, lakes and endless green plains. It has a hot climate all year, with humid summers, drier winters, seasonal floods and dense vegetation.'),
    ('Canada','Quebec','canada.jpg','Endless evergreen forests, beautiful lakes and rocky hills. Its boreal climate brings long, freezing winters and short, mild summers.'),
    ('Chile','Valparaíso','chile.jpg','Valparaíso combines the beauty of the Pacific coast with steep hills, cliffs and green valleys. Its Mediterranean climate means warm, dry summers and mild, rainy winters.'),
    ('India','Punjab','india.jpg','Punjab is a fertile region of wide, flat plains, golden fields and flowing rivers. It experiences extremely hot summers, a rainy monsoon season and cooler winters, with temperatures sometimes rising above 40°C.'),
]
cards=[]
for country,place,filename,description in gallery_specs:
    image_path=ROOT/'media'/filename
    if image_path.exists():
        image_data=base64.b64encode(image_path.read_bytes()).decode('ascii')
        cards.append(f'''<button class="flip-card" type="button" aria-pressed="false" aria-label="Show description of {place}">
          <span class="flip-inner"><span class="card-face card-front"><img src="data:image/jpeg;base64,{image_data}" alt="Landscape in {place}"><span class="card-label">{place}<small>{country}</small></span></span>
          <span class="card-face card-back"><span class="back-kicker">{place} · landscape & climate</span><span class="back-copy">{description}</span><span class="back-hint">Click to return to photograph</span></span></span>
        </button>''')
gallery_html='''<!doctype html><html><head><meta charset="utf-8"><style>
*{box-sizing:border-box}body{margin:0;background:transparent;color:#202723;font-family:Arial,Helvetica,sans-serif}.gallery{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}.flip-card{height:320px;padding:0;border:0;background:transparent;cursor:pointer;text-align:left;perspective:1200px;color:inherit}.flip-card:focus-visible{outline:3px solid #315a49;outline-offset:4px}.flip-inner{position:relative;display:block;width:100%;height:100%;transition:transform .65s cubic-bezier(.2,.7,.2,1);transform-style:preserve-3d}.flip-card.flipped .flip-inner{transform:rotateY(180deg)}.card-face{position:absolute;inset:0;display:block;overflow:hidden;border:1px solid #cbd2ca;border-radius:8px;backface-visibility:hidden;-webkit-backface-visibility:hidden}.card-front img{width:100%;height:100%;display:block;object-fit:cover}.card-front:after{content:"";position:absolute;inset:35% 0 0;background:linear-gradient(transparent,rgba(0,0,0,.72))}.card-label{position:absolute;z-index:1;left:22px;bottom:18px;color:white;font-family:Georgia,serif;font-size:1.6rem}.card-label small{display:block;margin-top:3px;font:700 .74rem Arial,sans-serif;letter-spacing:.11em;text-transform:uppercase}.card-back{display:flex;flex-direction:column;justify-content:center;gap:16px;padding:clamp(18px,3vw,32px);background:#e9eee8;transform:rotateY(180deg)}.back-kicker{color:#315a49;font-size:.76rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase}.back-copy{font-family:Georgia,serif;font-size:clamp(1rem,1.45vw,1.3rem);line-height:1.55}.back-hint{color:#515b55;font-size:.8rem}@media(max-width:650px){.gallery{grid-template-columns:1fr}.flip-card{height:290px}}
</style></head><body><div class="gallery">'''+''.join(cards)+'''</div><script>
document.querySelectorAll('.flip-card').forEach(card=>card.addEventListener('click',()=>{const wasFlipped=card.classList.contains('flipped');document.querySelectorAll('.flip-card').forEach(other=>{other.classList.remove('flipped');other.setAttribute('aria-pressed','false')});if(!wasFlipped){card.classList.add('flipped');card.setAttribute('aria-pressed','true')}}));
</script></body></html>'''
st.iframe(gallery_html,height='content',alt='Interactive gallery of the four study area landscapes and climate descriptions.')

st.markdown('<div id="method-validation" class="section-anchor"></div>',unsafe_allow_html=True)
st.markdown('## How the sensor comparison was evaluated')
st.caption('The default dashboard and API measure is Approximate daily 1 km clusters, which scored better on the reported 2020–2024 holdout. Count-aligned HFAI remains available for comparison and preserves the candidate chosen before that holdout.')
st.markdown(f'''Three candidate ways of bringing VIIRS toward a MODIS-equivalent monthly count were compared against MODIS. The selected candidate is **{report.get('selected_method','—').replace('_',' ')}**. A low error means closer agreement on this shared grid; it does not mean the instruments observe identical fires.''')
selection=report.get('method_selection',{})
selection_scores=pd.DataFrame(selection.get('selection_scores',[]))
st.markdown(f'''**Selection safeguard:** factors for method selection were fit on 2012–2017; the method was chosen on separate 2018–2019 data. The selected method (**{report.get('selected_method','—').replace('_',' ')}**) was refit on 2012–2019 and tested on the untouched 2020–2024 period.''')
if not selection_scores.empty:
    selection_scores['method']=selection_scores.method.str.replace('_',' ').str.title()
    st.caption('Method choice on the separate 2018–2019 selection period (pooled MAE in MODIS detections per cell-month):')
    st.dataframe(selection_scores[['method','n','mae','rmse','bias','r2']].rename(columns={'method':'Candidate','n':'Selection n','mae':'MAE','rmse':'RMSE','bias':'Bias','r2':'R²'}).style.format({'MAE':'{:.2f}','RMSE':'{:.2f}','Bias':'{:+.2f}','R²':'{:.3f}'}),hide_index=True,width='stretch')
scores=pd.DataFrame(report.get('candidate_scores',[]))
if not scores.empty:
    st.caption('Final 2020–2024 holdout diagnostics. The selected method was fixed before this period; rankings among all candidates here are descriptive and were not used to choose it.')
    scores['method']=scores.method.str.replace('_',' ').str.title()
    display=scores[['method','pooled_n','pooled_holdout_mae','pooled_holdout_rmse','pooled_holdout_bias','pooled_holdout_r2','macro_holdout_mae','macro_holdout_rmse','macro_holdout_bias','macro_holdout_r2','scored_sensor_region_pairs']].rename(columns={'method':'Candidate','pooled_n':'Pooled n','pooled_holdout_mae':'Pooled MAE','pooled_holdout_rmse':'Pooled RMSE','pooled_holdout_bias':'Pooled bias','pooled_holdout_r2':'Pooled R²','macro_holdout_mae':'Macro MAE','macro_holdout_rmse':'Macro RMSE','macro_holdout_bias':'Macro bias','macro_holdout_r2':'Macro R²','scored_sensor_region_pairs':'Available pairs'})
    st.dataframe(display.style.format({'Pooled MAE':'{:.2f}','Pooled RMSE':'{:.2f}','Pooled bias':'{:+.2f}','Pooled R²':'{:.3f}','Macro MAE':'{:.2f}','Macro RMSE':'{:.2f}','Macro bias':'{:+.2f}','Macro R²':'{:.3f}'}),hide_index=True,width='stretch')
    score_keys=scores.method.str.lower().str.replace(' ','_',regex=False)
    cluster_score=scores.loc[score_keys.eq('cluster_collapse')]
    ratio_score=scores.loc[score_keys.eq('count_ratio')]
    if not cluster_score.empty and not ratio_score.empty:
        c=cluster_score.iloc[0]; r=ratio_score.iloc[0]
        st.info(f"Default measure: **Cluster Collapse** (holdout MAE {c.pooled_holdout_mae:.2f}, R² {c.pooled_holdout_r2:.3f}); count-ratio HFAI scored MAE {r.pooled_holdout_mae:.2f}, R² {r.pooled_holdout_r2:.3f}. This default follows the observed holdout ranking; those scores are descriptive for this choice, not a new independent test.")
overlap_path=ROOT/'outputs'/'figures'/'overlap_before_after.png'
overlap_data_path=ROOT/'outputs'/'data'/'overlap_before_after.json'
if overlap_path.exists() and overlap_data_path.exists():
    overlap_result=json.loads(overlap_data_path.read_text(encoding='utf-8'))
    raw_metrics=overlap_result['pooled_before_raw_viirs']
    corrected_metrics=overlap_result['pooled_after_selected_method']
    st.markdown('**Same held-out overlap: raw versus corrected**')
    st.markdown(f'''On the same {raw_metrics['n']:,} sensor–region–cell-month comparisons from 2020–2024, raw VIIRS has MAE {raw_metrics['mae']:.2f}, RMSE {raw_metrics['rmse']:.2f}, bias {raw_metrics['bias']:+.2f} and R² {raw_metrics['r2']:.2f}. The selected {overlap_result['selected_method_label'].lower()} estimate gives MAE {corrected_metrics['mae']:.2f}, RMSE {corrected_metrics['rmse']:.2f}, bias {corrected_metrics['bias']:+.2f} and R² {corrected_metrics['r2']:.2f}; pooled MAE falls {overlap_result['mae_reduction_fraction']:.1%}.''')
    st.image(str(overlap_path),width='stretch')
st.caption('These diagnostics evaluate aggregate cell-month agreement with MODIS. They do not validate fire-event matching. The daily-cluster view is an additional repeat-detection proxy; its 1 km grouping is approximate and does not identify incidents.')
seam_path=ROOT/'outputs'/'figures'/'seam_2012_comparison.png'
if seam_path.exists():
    st.markdown('**2012 sensor-start seam check**')
    st.image(str(seam_path),width='stretch')
    st.caption('Raw detections rise as S-NPP enters the record; the adjusted HFAI step is smaller in Pantanal and Punjab. Year-to-year fire activity also changes, so this comparison does not isolate a causal sensor effect.')
with st.expander('How to read the index'):
    st.markdown('''**HFAI** means Harmonized Fire Activity Index. For each 0.25° cell and month, VIIRS detections are scaled into MODIS-equivalent units using a factor learned from that region’s shared-cell overlap. The displayed index is the mean of the available sensor estimates, so the value does not rise merely because a second VIIRS platform was added. MODIS-only months before VIIRS began remain in the calendar, but they cannot recover small or cool fires MODIS never detected.''')
    st.markdown('''The count-ratio candidate estimates a cell-month multiplier; the cluster candidate first collapses same-day/day-night VIIRS detections into approximate latitude-adjusted 1-km cells; the FRP candidate uses summed fire radiative power as a predictor of MODIS counts. A fixed 2012–2019 calibration is evaluated out of time in 2020–2024. NOAA-21 started too late for the fit and therefore receives a VIIRS Collection 2 family transfer factor, which is explicitly not an independent NOAA-21 training result.''')
    st.markdown('**Study scope:** all 16 requested historical source files are included across the four configured study areas. Results are limited to these rectangles; this is not a global data product.')
    st.markdown('''The confidence filter favors fewer false alarms: MODIS numerical confidence ≥30 keeps nominal/high detections, and VIIRS categorical confidence keeps nominal/high detections. This also excludes low-confidence pixels, which can include real fires; the two sensors use different confidence schemes and their labels are not directly comparable. The filtered record is therefore a quality choice, not a complete census.''')
    st.markdown('''A point marks the center of a detected satellite pixel, not a fire perimeter or exact ignition location. HFAI is intentionally defined over the supplied detection record and monthly grid: it compares aggregate activity in the same place and season. A month with no qualifying points means no qualifying detections in this record; it is not a claim that no fire occurred.''')
    st.markdown('''The project focuses on historical pattern comparison. Cloud-conditioned detection rates and cause labels answer separate questions and are not part of this index. Accordingly, do not read HFAI as an individual-fire match, ignition classifier, burned-area estimate, or forecast. FRP is reported in megawatts and is only a candidate predictor of MODIS counts; it is not converted to emissions or ground intensity.''')
    st.markdown('**Source filter:** archive `type = 0` is the FIRMS product-field filter; it does not identify agricultural burning, wildfire, or ignition cause, and some static non-fire sources may be mis-typed. NRT files without a `type` field are explicitly retained only after the standard-archive cutoff. **Recent-data rule:** NRT is provisional and can be revised.')

st.markdown('<div id="downloads" class="section-anchor"></div>',unsafe_allow_html=True)
st.markdown('## Download this selection')
st.markdown('Export the selected activity grid, calendar image, or one-page brief. The optional point CSV contains quality-filtered source detections, not reconstructed incidents.')
if st.session_state.applied_measure=='Approximate daily 1 km clusters':
    st.info('The grid, calendar and brief use the selected cluster proxy. The optional point CSV remains a row-level source-detection export with primary HFAI-equivalent weights; it is not a cluster/event export.')
export_frame=selected[['region','year','month','month_start','lat','lon','hfai','raw_sensor_detections','MODIS_count','SNPP_VIIRS_count','NOAA20_VIIRS_count','NOAA21_VIIRS_count','hfai_method']].copy() if not selected.empty else pd.DataFrame()
if not export_frame.empty:
    export_frame['activity_measure']=measure_label
    export_frame=export_frame.rename(columns={'hfai':'activity_value'})
    csv_bytes=export_frame.to_csv(index=False).encode('utf-8')
    p1,p2,p3=st.columns(3)
    with p1: st.download_button('Download harmonized cell-month CSV',data=csv_bytes,file_name=f'pyrochron_{region.lower()}_{start_iso}_{end_iso}_grid.csv',mime='text/csv',width='stretch')
    with p2:
        png=make_calendar_png(selected_cal,region,start_iso,end_iso,index_label=measure_label)
        st.download_button('Download calendar PNG',data=png,file_name=f'pyrochron_{region.lower()}_calendar.png',mime='image/png',width='stretch')
    summary={'hfai':selected_total,'raw_sensor_detections':int(selected.raw_sensor_detections.sum()),'grid_cells':selected[['cell_lat_i','cell_lon_i']].drop_duplicates().shape[0],
             'active_sensor_count':int(selected.active_sensor_count.max()),'hfai_month_median':expected_total,'hfai_vs_median_ratio':period_ratio}
    with p3:
        pdf=make_brief_pdf(region,start_iso,end_iso,summary,report,index_label=measure_label)
        st.download_button('Download one-page PDF brief',data=pdf,file_name=f'pyrochron_{region.lower()}_brief.pdf',mime='application/pdf',width='stretch')
    source_archives_available=any((ROOT/'archives').glob('fire_*.csv'))
    if source_archives_available and st.button('Prepare filtered NASA hotspot-point CSV',help='Scans the archived source rows, applies the same quality rules and includes each row’s HFAI-equivalent contribution.'):
        with st.spinner('Reading the selected NASA archive rows…'):
            payload,manifest=export_hotspots(region,bbox,start_iso,end_iso,report,months=st.session_state.applied_months)
            st.session_state['point_export']={'payload':payload,'manifest':manifest,'region':region,'start':start_iso,'end':end_iso}
    elif not source_archives_available:
        st.info('Custom point exports need the optional FIRMS archive CSVs. This deployment includes a representative Quebec May–August 2023 point sample and its exact sample manifest.')
        sample_path=ROOT/'outputs'/'data'/'selected_quebec_2023_source_points.csv.gz'
        sample_manifest_path=ROOT/'outputs'/'data'/'selected_quebec_2023_source_points_manifest.json'
        if sample_path.exists() and sample_manifest_path.exists():
            sample_cols=st.columns(2)
            with sample_cols[0]:
                st.download_button('Download Quebec demo point sample · May–Aug 2023',data=sample_path.read_bytes(),file_name='pyrochron_quebec_2023_sample_hotspots.csv.gz',mime='application/gzip',width='stretch')
            with sample_cols[1]:
                st.download_button('Download sample manifest JSON',data=sample_manifest_path.read_bytes(),file_name='pyrochron_quebec_2023_sample_manifest.json',mime='application/json',width='stretch')
    point=st.session_state.get('point_export')
    if point and point['region']==region and point['start']==start_iso and point['end']==end_iso:
        manifest=point['manifest']; exported=manifest['exported_rows']; matched=manifest['matched_rows']
        download_cols=st.columns(2)
        with download_cols[0]:
            st.download_button(f'Download filtered point CSV.gz · {exported:,} rows',data=point['payload'],file_name=f'pyrochron_{region.lower()}_{start_iso}_{end_iso}_hotspots.csv.gz',mime='application/gzip',width='stretch')
        with download_cols[1]:
            manifest_data=json.dumps(manifest,indent=2,sort_keys=True).encode('utf-8')
            st.download_button('Download point-export manifest JSON',data=manifest_data,file_name=f'pyrochron_{region.lower()}_{start_iso}_{end_iso}_point_manifest.json',mime='application/json',width='stretch')
        sensor_counts=', '.join(f'{sensor}: {counts["exported"]:,}/{counts["matched"]:,}' for sensor,counts in manifest['per_sensor'].items() if counts['matched'])
        if manifest['truncated']:
            st.caption(f'Deterministic proportional sample: {exported:,} exported from {matched:,} matched source rows (cap {manifest["row_cap"]:,}). This point file is a sample, not a complete count or census. The JSON manifest has exact per-sensor/month quotas. Per sensor, exported/matched: {sensor_counts}. HFAI-equivalent contributions retain the sensor calibration and active-sensor weighting.')
        else:
            st.caption(f'Complete filtered export: all {matched:,} matched source rows are included. Per sensor, exported/matched: {sensor_counts}. HFAI-equivalent contributions retain the sensor calibration and active-sensor weighting.')

st.markdown('<div id="api-access" class="section-anchor"></div>',unsafe_allow_html=True)
st.markdown('## Query monthly totals through the local API')
api_measure=active_measure
api_query=f'http://127.0.0.1:8765/api/calendar?bbox={bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}&from={start_iso}&to={end_iso}&region={region}&measure={api_measure}'
st.code(f'GET {api_query}',language='text')
api_link,health_link=st.columns([1,1])
with api_link: st.link_button('Open JSON response',api_query,use_container_width=True)
with health_link: st.link_button('Check API health','http://127.0.0.1:8765/healthz',use_container_width=True)
st.caption(f'The API link uses the selected {st.session_state.applied_measure} measure. Run `launch.ps1` to start both services locally, or `docker compose up --build` to run the API and dashboard in containers. Historical API requests read local data; `/api/roi` queries NASA FIRMS NRT and needs a backend-only `FIRMS_MAP_KEY`.')
st.download_button('Download bibliography and data sources',data=(ROOT/'bibliography.txt').read_bytes(),
                   file_name='pyrochron_bibliography.txt',mime='text/plain',width='content')

st.markdown(f'<footer class="site-footer"><strong>Pyrochron</strong> · NASA FIRMS MODIS C6.1 and VIIRS Collection 2 archive exports · NASA POWER MERRA-2 monthly context · retrieved 12 November 2026 · {len(REGIONS)} configured study areas · active measure: {active_measure.replace("_"," ")}</footer>',unsafe_allow_html=True)
