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
    # Read the uploaded file line by line
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
        
        # Get substances (columns with STABSPECTRO)
        cols = [c for c in matrix_df.columns if 'STABSPECTRO' in c and not c.startswith('s')]
        sel_sub = st.sidebar.selectbox("Substance (e.g. K40)", cols)

        st.sidebar.subheader("Precision Filters")
        lat_min, lat_max = float(matrix_df['GPS_0020_Lat'].min()), float(matrix_df['GPS_0020_Lat'].max())
        lon_min, lon_max = float(matrix_df['GPS_0020_Lon'].min()), float(matrix_df['GPS_0020_Lon'].max())
        
        lat_range = st.sidebar.slider("Lat", lat_min, lat_max, (lat_min, lat_max), format="%.5f")
        lon_range = st.sidebar.slider("Lon", lon_min, lon_max, (lon_min, lon_max), format="%.5f")
        h_range = st.sidebar.slider("Height (m)", 0, 100, (0, 60))
        
        st.sidebar.subheader("Map Settings")
        grid_res = st.sidebar.slider("Grid Resolution (approx meters)", 0.1, 5.0, 1.0, 0.1)

        # --- APPLY FILTERS ---
        df_f = matrix_df[
            (matrix_df['GPS_0020_Lat'].between(lat_range[0], lat_range[1])) &
            (matrix_df['GPS_0020_Lon'].between(lon_range[0], lon_range[1])) &
            (matrix_df['GPS_0020_Height'].between(h_range[0], h_range[1]))
        ].copy()

        plot_df = df_f.dropna(subset=['GPS_0020_Lat', 'GPS_0020_Lon', sel_sub])

        # --- MAIN TABS ---
        st.title("⚰️ Archaeological Survey Dashboard")
        t1, t2, t3, t4, t5 = st.tabs(["🗺️ Raw Map", "⬢ Hexbin Density", "⬛ Square Grid", "🔥 Fluid (Interpolated)", "📈 3D View"])

        # ESRI Satellite Tile Source
        esri_satellite = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"

        with t1:
            st.subheader("Direct Measurements")
            fig_raw = px.scatter_map(plot_df, lat='GPS_0020_Lat', lon='GPS_0020_Lon', color=sel_sub, size_max=10, zoom=18, color_continuous_scale="Viridis", height=700)
            fig_raw.update_layout(map_style="white-bg", map_layers=[{"below": 'traces', "sourcetype": "raster", "source": [esri_satellite]}])
            st.plotly_chart(fig_raw, use_container_width=True)

        with t2:
            st.subheader("Hexbin Density (Precise)")
            # Using density_map is the modern replacement for hexbinning in Plotly maps
            fig_hex = px.density_map(plot_df, lat='GPS_0020_Lat', lon='GPS_0020_Lon', z=sel_sub, radius=15, zoom=18, color_continuous_scale="Viridis", height=700)
            fig_hex.update_layout(map_style="white-bg", map_layers=[{"below": 'traces', "sourcetype": "raster", "source": [esri_satellite]}])
            st.plotly_chart(fig_hex, use_container_width=True)

        with t3:
            st.subheader("Square Grid Aggregation")
            # Rounding coordinates to create a grid. 0.00001 lat is roughly 1.1 meters.
            step = 0.00001 * grid_res
            df_grid = plot_df.copy()
            df_grid['Lat_G'] = (df_grid['GPS_0020_Lat'] / step).round() * step
            df_grid['Lon_G'] = (df_grid['GPS_0020_Lon'] / step).round() * step
            df_grid = df_grid.groupby(['Lat_G', 'Lon_G'])[sel_sub].mean().reset_index()
            
            fig_sq = px.scatter_map(df_grid, lat='Lat_G', lon='Lon_G', color=sel_sub, size_max=12, zoom=18, color_continuous_scale="Viridis", height=700)
            fig_sq.update_layout(map_style="white-bg", map_layers=[{"below": 'traces', "sourcetype": "raster", "source": [esri_satellite]}])
            st.plotly_chart(fig_sq, use_container_width=True)

        with t4:
            st.subheader("Interpolated Fluid Heatmap")
            if len(plot_df) > 10:
                res = 150 # Grid resolution
                x, y, z = plot_df['GPS_0020_Lon'].values, plot_df['GPS_0020_Lat'].values, plot_df[sel_sub].values
                grid_x, grid_y = np.mgrid[x.min():x.max():complex(res), y.min():y.max():complex(res)]
                grid_z = griddata((x, y), z, (grid_x, grid_y), method='linear')
                grid_z = gaussian_filter(grid_z, sigma=1.5) # Slight blur for fluidity
                
                # Flatten back to dataframe for plotting
                df_interp = pd.DataFrame({'Lon': grid_x.flatten(), 'Lat': grid_y.flatten(), 'Val': grid_z.flatten()}).dropna()
                fig_int = px.density_map(df_interp, lat='Lat', lon='Lon', z='Val', radius=5, zoom=18, color_continuous_scale="Viridis", height=700)
                fig_int.update_layout(map_style="white-bg", map_layers=[{"below": 'traces', "sourcetype": "raster", "source": [esri_satellite]}])
                st.plotly_chart(fig_int, use_container_width=True)
            else:
                st.info("Insufficient data points for interpolation.")

        with t5:
            st.subheader("3D Concentration Profiles")
            fig_3d = px.scatter_3d(plot_df, x='GPS_0020_Lon', y='GPS_0020_Lat', z=sel_sub, color=sel_sub, color_continuous_scale="Viridis", height=700)
            st.plotly_chart(fig_3d, use_container_width=True)

else:
    st.info("👋 Please upload a .json file in the sidebar to begin.")
