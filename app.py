import streamlit as st
import json
import pandas as pd
import plotly.express as px
import numpy as np
import os
from streamlit_plotly_events import plotly_events

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Grave Analysis Pro", layout="wide")

# Initialize Session State
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

# --- SIDEBAR ---
st.sidebar.title("🛠️ Precise Control Panel")
project_name = st.sidebar.text_input("Project Name", value="New Survey Site")

st.sidebar.divider()
# Switch between moving and drawing
map_mode = st.sidebar.radio("Map Mouse Mode", ["Navigate (Zoom/Pan)", "Select (Lasso/Box)"])

if st.sidebar.button("🔄 Reset All Manual Exclusions"):
    st.session_state.excluded_ids = set()
    st.rerun()

uploaded_file = st.sidebar.file_uploader("Upload .json file", type=["json"])

if uploaded_file is not None:
    matrix_df = load_and_process_data(uploaded_file)
    
    if matrix_df is not None:
        cols = [c for c in matrix_df.columns if 'STABSPECTRO' in c and not c.startswith('s')]
        sel_sub = st.sidebar.selectbox("Analyze Substance", cols)

        # Apply Filtering Logic
        df_f = matrix_df.copy()
        df_f['is_excluded'] = df_f['point_id'].isin(st.session_state.excluded_ids)
        active_df = df_f[~df_f['is_excluded']].dropna(subset=[sel_sub])

        # --- MAIN DASHBOARD ---
        st.title(f"⚰️ {project_name}")
        
        t1, t2, t3 = st.tabs(["🗺️ Selection Map", "📈 3D View", "📋 Data"])

        with t1:
            if map_mode == "Select (Lasso/Box)":
                st.warning("✨ **Lasso Active**: Just click and drag on the map to exclude points.")
            
            # Mapbox is required for stable selection tools
            fig_map = px.scatter_mapbox(
                df_f, lat='GPS_0020_Lat', lon='GPS_0020_Lon',
                color=sel_sub, 
                opacity=df_f['is_excluded'].map({True: 0.15, False: 0.9}),
                size_max=12, zoom=18,
                color_continuous_scale="Viridis", height=750,
                hover_data=['point_id']
            )

            # Determine drag mode based on sidebar selection
            active_drag_mode = 'lasso' if map_mode == "Select (Lasso/Box)" else 'pan'
            
            fig_map.update_layout(
                mapbox_style="white-bg",
                mapbox_layers=[{
                    "below": 'traces', "sourcetype": "raster",
                    "source": ["https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"]
                }],
                margin={"r":0,"t":0,"l":0,"b":0},
                dragmode=active_drag_mode,
                modebar_visible=True, # FORCES toolbar to show
                clickmode='event+select'
            )

            # Define the configuration to FORCE the display of the modebar
            map_config = {
                'displayModeBar': True, # ALWAYS show the bar
                'modeBarButtonsToAdd': ['lasso2d', 'select2d'], # Explicitly add Lasso and Box
                'displaylogo': False
            }

            # Render map with custom configuration
            selected_points = plotly_events(
                fig_map, 
                select_event=True, 
                click_event=True, 
                key=f"map_{map_mode}", 
                override_height=750,
                override_width='100%'
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
            st.subheader("Data Matrix")
            st.dataframe(active_df, use_container_width=True)

else:
    st.info("👋 Upload a .json file to start.")
