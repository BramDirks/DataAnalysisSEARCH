import streamlit as st
import json
import pandas as pd
import plotly.express as px
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Sensor Data Analysis", layout="wide")

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

# --- SIDEBAR GUI ---
st.sidebar.title("🛠️ Control Panel")
uploaded_file = st.sidebar.file_uploader("Upload your .json data file", type=["json"])

if uploaded_file is not None:
    matrix_df = load_and_process_data(uploaded_file)
    if matrix_df is not None:
        st.sidebar.success("Data Loaded!")
        cols = [c for c in matrix_df.columns if 'STABSPECTRO' in c and not c.startswith('s')]
        sel_sub = st.sidebar.selectbox("Analyze Substance", cols)

        st.sidebar.subheader("Filters")
        lat_range = st.sidebar.slider("Lat Range", 50.0, 54.0, (matrix_df['GPS_0020_Lat'].min(), matrix_df['GPS_0020_Lat'].max()), step=0.0001)
        lon_range = st.sidebar.slider("Lon Range", 3.0, 8.0, (matrix_df['GPS_0020_Lon'].min(), matrix_df['GPS_0020_Lon'].max()), step=0.0001)
        
        use_h = st.sidebar.checkbox("Filter Height", value=True)
        h_range = st.sidebar.slider("Height (m)", 0, 100, (0, 55))
        
        # Heatmap specific controls
        st.sidebar.subheader("Heatmap Settings")
        blur = st.sidebar.slider("Smoothing (Blur)", 0.0, 5.0, 1.5)
        res = st.sidebar.slider("Resolution", 50, 300, 150)

        # --- APPLY FILTERS ---
        df_f = matrix_df[
            (matrix_df['GPS_0020_Lat'].between(lat_range[0], lat_range[1])) &
            (matrix_df['GPS_0020_Lon'].between(lon_range[0], lon_range[1]))
        ].copy()
        if use_h and 'GPS_0020_Height' in df_f.columns:
            df_f = df_f[df_f['GPS_0020_Height'].between(h_range[0], h_range[1])]

        plot_df = df_f.dropna(subset=['GPS_0020_Lat', 'GPS_0020_Lon', sel_sub])

        st.title("Concentration Survey Dashboard")
        t1, t2, t3, t4 = st.tabs(["🗺️ Map", "🔥 Interpolated Heatmap", "📈 3D View", "📋 Table"])

        with t1:
            if not plot_df.empty:
                fig_map = px.scatter_map(plot_df, lat='GPS_0020_Lat', lon='GPS_0020_Lon', color=sel_sub, size_max=15, zoom=14, color_continuous_scale="Viridis", height=600)
                fig_map.update_layout(map_style="white-bg", map_layers=[{"below": 'traces', "sourcetype": "raster", "source": ["https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"]}])
                st.plotly_chart(fig_map, use_container_width=True)

        with t2:
            st.subheader("Fluid Heatmap (Linear Interpolation)")
            if len(plot_df) > 10:
                # Prepare grid
                x = plot_df['GPS_0020_Lon'].values
                y = plot_df['GPS_0020_Lat'].values
                z = plot_df[sel_sub].values
                
                grid_x, grid_y = np.mgrid[x.min():x.max():complex(res), y.min():y.max():complex(res)]
                
                # Interpolate
                grid_z = griddata((x, y), z, (grid_x, grid_y), method='linear')
                
                # Smooth
                if blur > 0:
                    grid_z = gaussian_filter(grid_z, sigma=blur)

                # Convert grid back to long-form for Plotly
                grid_df = pd.DataFrame({
                    'Lon': grid_x.flatten(),
                    'Lat': grid_y.flatten(),
                    'Conc': grid_z.flatten()
                }).dropna()

                fig_heat = px.density_mapbox(grid_df, lat='Lat', lon='Lon', z='Conc', radius=10,
                                             center={"lat": y.mean(), "lon": x.mean()}, zoom=14,
                                             mapbox_style="stamen-terrain", height=600,
                                             color_continuous_scale="Viridis")
                # Add satellite back
                fig_heat.update_layout(mapbox_style="white-bg", map_layers=[{"below": 'traces', "sourcetype": "raster", "source": ["https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"]}])
                st.plotly_chart(fig_heat, use_container_width=True)
            else:
                st.info("Need more data points to interpolate.")

        with t3:
            if not plot_df.empty:
                fig_3d = px.scatter_3d(plot_df, x='GPS_0020_Lon', y='GPS_0020_Lat', z=sel_sub, color=sel_sub, color_continuous_scale="Viridis", height=600)
                st.plotly_chart(fig_3d, use_container_width=True)

        with t4:
            st.dataframe(plot_df)
            st.download_button("Export CSV", plot_df.to_csv(index=False), "data.csv")
else:
    st.info("👋 Please upload a .json file in the sidebar to begin.")
