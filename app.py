import streamlit as st
import json
import pandas as pd
import plotly.express as px
import numpy as np
import os
from streamlit_plotly_events import plotly_events

# --- PAGE CONFIGURATION ---
pd.set_option("styler.render.max_elements", 500000) 
st.set_page_config(page_title="High-Precision Grave Analysis", layout="wide")

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
    # Create a unique ID for every point based on its row index
    df['point_id'] = df.index
    return df

# --- SIDEBAR GUI ---
st.sidebar.title("🛠️ Precise Control Panel")

if st.sidebar.button("🔄 Reset Manual Exclusions"):
    st.session_state.excluded_ids = set()
    st.rerun()

uploaded_file = st.sidebar.file_uploader("Upload .json data file", type=["json"])

if uploaded_file is not None:
    matrix_df = load_and_process_data(uploaded_file)
    
    if matrix_df is not None:
        st.sidebar.success("Data Loaded!")
        
        # 1. Substance Selection
        cols = [c for c in matrix_df.columns if 'STABSPECTRO' in c and not c.startswith('s')]
        sel_sub = st.sidebar.selectbox("Analyze Substance", cols)

        # 2. Filtering Controls
        st.sidebar.subheader("📍 Precise Coordinates")
        in_lat_min = st.sidebar.number_input("Min Lat", value=float(matrix_df['GPS_0020_Lat'].min()), format="%.6f")
        in_lat_max = st.sidebar.number_input("Max Lat", value=float(matrix_df['GPS_0020_Lat'].max()), format="%.6f")
        in_lon_min = st.sidebar.number_input("Min Lon", value=float(matrix_df['GPS_0020_Lon'].min()), format="%.6f")
        in_lon_max = st.sidebar.number_input("Max Lon", value=float(matrix_df['GPS_0020_Lon'].max()), format="%.6f")

        # --- APPLY DATA FILTERS ---
        df_f = matrix_df.copy()
        
        # 1. Coordinate Filter
        df_f = df_f[df_f['GPS_0020_Lat'].between(in_lat_min, in_lat_max)]
        df_f = df_f[df_f['GPS_0020_Lon'].between(in_lon_min, in_lon_max)]
        
        # 2. Manual Exclusion Filter (Apply the Blacklist)
        df_f['is_excluded'] = df_f['point_id'].isin(st.session_state.excluded_ids)
        
        # Active dataframe for calculations/plots
        active_df = df_f[~df_f['is_excluded']].dropna(subset=[sel_sub])

        # --- DASHBOARD ---
        st.title("⚰️ Grave Detection & Analysis Dashboard")
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Points Found", len(active_df))
        m2.metric("Avg Conc", f"{active_df[sel_sub].mean():.3f}" if not active_df.empty else "0")
        m3.metric("Manually Excluded", len(st.session_state.excluded_ids))

        t1, t2, t3 = st.tabs(["🗺️ Selection Map", "📈 3D View", "📋 Raw Data"])

        with t1:
            st.info("💡 Use the **Lasso Select** or **Box Select** tools (top right of map) to exclude noise. Excluded points turn grey.")
            
            # Map shows all points, but fades the excluded ones
            fig_map = px.scatter_map(
                df_f, lat='GPS_0020_Lat', lon='GPS_0020_Lon',
                color=sel_sub, 
                opacity=df_f['is_excluded'].map({True: 0.1, False: 0.9}),
                size_max=12, zoom=19,
                color_continuous_scale="Viridis", height=750,
                hover_data=['point_id']
            )
            fig_map.update_layout(
                map_style="white-bg",
                map_layers=[{"below": 'traces', "sourcetype": "raster", "source": ["https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"]}],
                clickmode='event+select'
            )
            
            # Use plotly_events to capture the selection
            selected_points = plotly_events(fig_map, select_event=True, key="map_select")

            if selected_points:
                # Find the point_ids of the selected points and add to session state
                selected_ids = [df_f.iloc[p['pointIndex']]['point_id'] for p in selected_points]
                st.session_state.excluded_ids.update(selected_ids)
                st.rerun()

        with t2:
            if not active_df.empty:
                fig_3d = px.scatter_3d(active_df, x='GPS_0020_Lon', y='GPS_0020_Lat', z=sel_sub, color=sel_sub, height=750)
                st.plotly_chart(fig_3d, use_container_width=True)

        with t3:
            st.subheader("Data Table (Excluded points in Red)")
            
            def highlight_excluded(row):
                if row['is_excluded']:
                    return ['color: red; font-weight: bold'] * len(row)
                return [''] * len(row)

            # Styled dataframe showing excluded rows in red
            st.dataframe(
                df_f.style.apply(highlight_excluded, axis=1),
                use_container_width=True
            )
            st.download_button("Export Clean CSV", active_df.to_csv(index=False), "cleaned_survey.csv")

else:
    st.info("👋 Please upload your .json file.")
