import streamlit as st
import json
import pandas as pd
import plotly.express as px
import numpy as np
import os
from streamlit_plotly_events import plotly_events

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Grave Analysis Dashboard", layout="wide")

# Initialize Session State for manual exclusions
if 'excluded_ids' not in st.session_state:
    st.session_state.excluded_ids = set()

@st.cache_data
def load_and_process_data(uploaded_file):
    all_data = []
    uploaded_file.seek(0)
    for line in uploaded_file:
        line = line.decode("utf-8").strip()
        if not line: continue
        try:
            sensors = json.loads(line)
            for sensor in sensors:
                if 'eID' in sensor and 'v' in sensor:
                    row = {'vT': sensor['vT'], 'eID': sensor['eID']}
                    val = sensor['v']
                    if isinstance(val, dict):
                        for k, v in val.items():
                            row[f"{sensor['eID']}_{k}"] = v
                    else:
                        row[f"{sensor['eID']}_val"] = val
                    all_data.append(row)
        except: continue
    
    if not all_data: return None
    df = pd.DataFrame(all_data)
    df['Time_Sec'] = (df['vT'] / 1000).round().astype(int)
    df = df.groupby('Time_Sec').first().reset_index()
    df['point_id'] = df.index
    return df

# --- SIDEBAR GUI ---
st.sidebar.title("🛠️ Precise Control Panel")
project_name = st.sidebar.text_input("Project Name", value="New Survey Site")

st.sidebar.divider()
# --- THE EDIT MODE TOGGLE ---
edit_mode = st.sidebar.toggle("✏️ Lasso Edit Mode", help="Turn this ON to draw and exclude points. Turn OFF to zoom/pan the map.")

if st.sidebar.button("🔄 Reset Manual Exclusions"):
    st.session_state.excluded_ids = set()
    st.rerun()

uploaded_file = st.sidebar.file_uploader("Upload .json data file", type=["json"])

if uploaded_file is not None:
    matrix_df = load_and_process_data(uploaded_file)
    
    if matrix_df is not None:
        cols = [c for c in matrix_df.columns if 'STABSPECTRO' in c and not c.startswith('s')]
        sel_sub = st.sidebar.selectbox("Analyze Substance", cols)

        # Filters
        st.sidebar.subheader("📍 Precise Coordinates")
        in_lat_min = st.sidebar.number_input("Min Lat", value=float(matrix_df['GPS_0020_Lat'].min()), format="%.6f")
        in_lat_max = st.sidebar.number_input("Max Lat", value=float(matrix_df['GPS_0020_Lat'].max()), format="%.6f")
        in_lon_min = st.sidebar.number_input("Min Lon", value=float(matrix_df['GPS_0020_Lon'].min()), format="%.6f")
        in_lon_max = st.sidebar.number_input("Max Lon", value=float(matrix_df['GPS_0020_Lon'].max()), format="%.6f")

        # Data Logic
        df_f = matrix_df.copy()
        df_f = df_f[df_f['GPS_0020_Lat'].between(in_lat_min, in_lat_max)]
        df_f = df_f[df_f['GPS_0020_Lon'].between(in_lon_min, in_lon_max)]
        df_f['is_excluded'] = df_f['point_id'].isin(st.session_state.excluded_ids)
        active_df = df_f[~df_f['is_excluded']].dropna(subset=[sel_sub])

        st.title(f"⚰️ {project_name}")
        t1, t2, t3 = st.tabs(["🗺️ Selection Map", "📈 3D View", "📋 Raw Data"])

        with t1:
            if edit_mode:
                st.warning("✨ **Lasso Active**: Draw shapes on the map to exclude noise.")
            else:
                st.info("🔎 **Navigate Mode**: Move and zoom normally. Toggle 'Edit Mode' in the sidebar to exclude points.")

            # Mapbox for stable tools
            fig_map = px.scatter_mapbox(
                df_f, lat='GPS_0020_Lat', lon='GPS_0020_Lon',
                color=sel_sub, 
                opacity=df_f['is_excluded'].map({True: 0.1, False: 0.9}),
                size_max=12, zoom=18,
                color_continuous_scale="Viridis", height=750,
                hover_data=['point_id']
            )

            # Force Lasso when toggle is ON, otherwise allow Pan
            current_drag = 'lasso' if edit_mode else 'pan'

            fig_map.update_layout(
                mapbox_style="white-bg",
                mapbox_layers=[{
                    "below": 'traces', "sourcetype": "raster",
                    "source": ["https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"]
                }],
                margin={"r":0,"t":0,"l":0,"b":0},
                dragmode=current_drag,
                modebar_visible=True,
                clickmode='event+select'
            )

            # config to force tool visibility
            map_config = {'displayModeBar': True, 'modeBarButtonsToAdd': ['lasso2d', 'select2d']}

            selected_points = plotly_events(
                fig_map, 
                select_event=True, 
                click_event=True, 
                key=f"map_select_{edit_mode}", # Key changes to force refresh
                override_height=750
            )

            if selected_points:
                selected_ids = [df_f.iloc[p['pointIndex']]['point_id'] for p in selected_points]
                st.session_state.excluded_ids.update(selected_ids)
                st.rerun()

        with t2:
            if not active_df.empty:
                fig_3d = px.scatter_3d(active_df, x='GPS_0020_Lon', y='GPS_0020_Lat', z=sel_sub, color=sel_sub, height=750)
                st.plotly_chart(fig_3d, use_container_width=True)

        with t3:
            st.dataframe(active_df, use_container_width=True)

else:
    st.info("👋 Upload a .json file to begin.")
