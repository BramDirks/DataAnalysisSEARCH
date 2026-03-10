import streamlit as st
import json
import pandas as pd
import plotly.express as px
import numpy as np
import os
from streamlit_plotly_events import plotly_events

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Grave Analysis Dashboard", layout="wide")

# Initialize Session State for manual exclusions if it doesn't exist
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
    # Unique ID for tracking manual selection
    df['point_id'] = df.index
    return df

# --- SIDEBAR GUI ---
st.sidebar.title("🛠️ Precise Control Panel")

# Project Management
project_name = st.sidebar.text_input("Project Name", value="New Survey Site")

if st.sidebar.button("🔄 Reset All Manual Exclusions"):
    st.session_state.excluded_ids = set()
    st.rerun()

uploaded_file = st.sidebar.file_uploader("Upload .json data file", type=["json"])

if uploaded_file is not None:
    matrix_df = load_and_process_data(uploaded_file)
    
    if matrix_df is not None:
        st.sidebar.success("Data Loaded Successfully!")
        
        cols = [c for c in matrix_df.columns if 'STABSPECTRO' in c and not c.startswith('s')]
        sel_sub = st.sidebar.selectbox("Analyze Substance", cols)

        st.sidebar.divider()

        # High-Precision Inputs
        st.sidebar.subheader("📍 Area of Interest")
        in_lat_min = st.sidebar.number_input("Min Latitude", value=float(matrix_df['GPS_0020_Lat'].min()), format="%.6f")
        in_lat_max = st.sidebar.number_input("Max Latitude", value=float(matrix_df['GPS_0020_Lat'].max()), format="%.6f")
        in_lon_min = st.sidebar.number_input("Min Longitude", value=float(matrix_df['GPS_0020_Lon'].min()), format="%.6f")
        in_lon_max = st.sidebar.number_input("Max Longitude", value=float(matrix_df['GPS_0020_Lon'].max()), format="%.6f")

        st.sidebar.divider()
        p_range = st.sidebar.slider("Noise Filter (Percentile)", 0, 100, (0, 100))

        # --- DATA FILTERING LOGIC ---
        df_f = matrix_df.copy()
        # 1. Geo Filter
        df_f = df_f[df_f['GPS_0020_Lat'].between(in_lat_min, in_lat_max)]
        df_f = df_f[df_f['GPS_0020_Lon'].between(in_lon_min, in_lon_max)]
        
        # 2. Identify Manually Excluded
        df_f['is_excluded'] = df_f['point_id'].isin(st.session_state.excluded_ids)
        
        # 3. Active Dataset (Excludes the manual selections)
        active_df = df_f[~df_f['is_excluded']].dropna(subset=[sel_sub])

        # --- DASHBOARD ---
        st.title(f"⚰️ {project_name}")
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Points Found", len(active_df))
        m2.metric("Avg Concentration", f"{active_df[sel_sub].mean():.3f}" if not active_df.empty else "0")
        m3.metric("Manually Excluded", len(st.session_state.excluded_ids))

        t1, t2, t3 = st.tabs(["🗺️ Selection Map", "📈 3D View", "📋 Raw Data"])

        with t1:
            st.info("💡 **How to exclude:** Use the **Box Select** or **Lasso Select** tools in the map toolbar. Click individual points to exclude them.")
            
            # Map shows all points, but fades the excluded ones
            fig_map = px.scatter_map(
                df_f, lat='GPS_0020_Lat', lon='GPS_0020_Lon',
                color=sel_sub, 
                opacity=df_f['is_excluded'].map({True: 0.15, False: 0.9}),
                size_max=12, zoom=19,
                color_continuous_scale="Viridis", height=750,
                hover_data=['point_id']
            )
            fig_map.update_layout(
                map_style="white-bg",
                map_layers=[{"below": 'traces', "sourcetype": "raster", "source": ["https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"]}],
                clickmode='event+select'
            )
            
            # Capture selection events (This replaces the standard st.plotly_chart)
            selected_points = plotly_events(fig_map, select_event=True, click_event=True, key="map_select")

            if selected_points:
                # Find the IDs of the selected/clicked points
                selected_ids = [df_f.iloc[p['pointIndex']]['point_id'] for p in selected_points]
                st.session_state.excluded_ids.update(selected_ids)
                st.rerun()

        with t2:
            if not active_df.empty:
                fig_3d = px.scatter_3d(active_df, x='GPS_0020_Lon', y='GPS_0020_Lat', z=sel_sub, color=sel_sub, height=750)
                st.plotly_chart(fig_3d, use_container_width=True)

        with t3:
            st.subheader("Matrix View")
            st.dataframe(active_df, use_container_width=True)
            csv_data = active_df.to_csv(index=False).encode('utf-8')
            st.download_button("Export Clean CSV", csv_data, f"{project_name}_clean.csv")

else:
    st.info("👋 Please upload your .json file.")
