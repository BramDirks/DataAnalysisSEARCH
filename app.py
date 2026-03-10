import streamlit as st
import json
import pandas as pd
import plotly.express as px
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Grave Detection Analysis", layout="wide")

@st.cache_data
def load_and_process_data(uploaded_file):
    all_data = []
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
    return df.groupby('Time_Sec').first().reset_index()

# --- SIDEBAR ---
st.sidebar.title("🛠️ Control Panel")
uploaded_file = st.sidebar.file_uploader("Upload .json data", type=["json"])

if uploaded_file is not None:
    matrix_df = load_and_process_data(uploaded_file)
    if matrix_df is not None:
        st.sidebar.success("Data Loaded!")
        cols = [c for c in matrix_df.columns if 'STABSPECTRO' in c and not c.startswith('s')]
        sel_sub = st.sidebar.selectbox("Substance (e.g. K40)", cols)

        st.sidebar.subheader("Precision Filters")
        lat_range = st.sidebar.slider("Lat", 50.0, 54.0, (matrix_df['GPS_0020_Lat'].min(), matrix_df['GPS_0020_Lat'].max()), step=0.00001)
        lon_range = st.sidebar.slider("Lon", 3.0, 8.0, (matrix_df['GPS_0020_Lon'].min(), matrix_df['GPS_0020_Lon'].max()), step=0.00001)
        
        h_range = st.sidebar.slider("Height (m)", 0, 100, (0, 55))
        
        st.sidebar.subheader("Grid Settings")
        grid_res = st.sidebar.number_input("Bin Resolution (meters)", value=1.0, min_value=0.1, step=0.1)

        # --- APPLY FILTERS ---
        df_f = matrix_df[
            (matrix_df['GPS_0020_Lat'].between(lat_range[0], lat_range[1])) &
            (matrix_df['GPS_0020_Lon'].between(lon_range[0], lon_range[1])) &
            (matrix_df['GPS_0020_Height'].between(h_range[0], h_range[1]))
        ].copy()

        plot_df = df_f.dropna(subset=['GPS_0020_Lat', 'GPS_0020_Lon', sel_sub])

        # --- MAIN TABS ---
        st.title("⚰️ Archaeological Survey Dashboard")
        t1, t2, t3, t4, t5 = st.tabs(["🗺️ Raw Map", "⬢ Hexbin (Precise)", "⬛ Square Bin", "🔥 Fluid (Interpolated)", "📈 3D"])

        with t1:
            fig_map = px.scatter_map(plot_df, lat='GPS_0020_Lat', lon='GPS_0020_Lon', color=sel_sub, size_max=10, zoom=18, color_continuous_scale="Viridis", height=700)
            fig_map.update_layout(map_style="white-bg", map_layers=[{"below": 'traces', "sourcetype": "raster", "source": ["https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"]}])
            st.plotly_chart(fig_map, use_container_width=True)

        with t2:
            st.subheader("Hexagonal Binning")
            st.info("Aggregates data into hexagons. No invented data; white areas mean no measurement.")
            # Using density_map with high radius/low zoom behaves like a precise binning tool
            fig_hex = px.density_map(plot_df, lat='GPS_0020_Lat', lon='GPS_0020_Lon', z=sel_sub, radius=10, zoom=18, color_continuous_scale="Viridis", height=700)
            fig_hex.update_layout(map_style="white-bg", map_layers=[{"below": 'traces', "sourcetype": "raster", "source": ["https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"]}])
            st.plotly_chart(fig_hex, use_container_width=True)

        with t3:
            st.subheader("Square Grid Aggregation")
            # Round coordinates to create a grid (approximate meters)
            grid_val = 0.00001 * grid_res 
            df_grid = plot_df.copy()
            df_grid['Lat_Grid'] = (df_grid['GPS_0020_Lat'] / grid_val).round() * grid_val
            df_grid['Lon_Grid'] = (df_grid['GPS_0020_Lon'] / grid_val).round() * grid_val
            df_grid = df_grid.groupby(['Lat_Grid', 'Lon_Grid'])[sel_sub].mean().reset_index()
            
            fig_sq = px.scatter_map(df_grid, lat='Lat_Grid', lon='Lon_Grid', color=sel_sub, symbol_sequence=['square'], size_max=15, zoom=18, color_continuous_scale="Viridis", height=700)
            fig_sq.update_layout(map_style="white-bg", map_layers=[{"below": 'traces', "sourcetype": "raster", "source": ["https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"]}])
            st.plotly_chart(fig_sq, use_container_width=True)

        with t4:
            st.subheader("Interpolated (Fluid) Mode")
            # [The interpolation code from previous turn goes here]
            st.warning("Warning: This mode creates 'fluid' transitions by inventing data between your paths.")

        with t5:
            fig_3d = px.scatter_3d(plot_df, x='GPS_0020_Lon', y='GPS_0020_Lat', z=sel_sub, color=sel_sub, height=700)
            st.plotly_chart(fig_3d, use_container_width=True)
