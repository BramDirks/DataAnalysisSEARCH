import streamlit as st
import json
import pandas as pd
import plotly.express as px
import numpy as np
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="High-Precision Grave Analysis", layout="wide")

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
st.sidebar.title("🛠️ Precise Control Panel")
uploaded_file = st.sidebar.file_uploader("Upload .json data file", type=["json"])

if uploaded_file is not None:
    matrix_df = load_and_process_data(uploaded_file)
    
    if matrix_df is not None:
        st.sidebar.success("Data Loaded!")
        
        # 1. Substance Selection
        cols = [c for c in matrix_df.columns if 'STABSPECTRO' in c and not c.startswith('s')]
        sel_sub = st.sidebar.selectbox("Analyze Substance", cols)

        st.sidebar.divider()

        # 2. High-Precision Numerical Inputs (replacing sliders)
        st.sidebar.subheader("📍 Precise Latitude")
        min_lat_val = float(matrix_df['GPS_0020_Lat'].min())
        max_lat_val = float(matrix_df['GPS_0020_Lat'].max())
        in_lat_min = st.sidebar.number_input("Min Latitude", value=min_lat_val, format="%.6f", step=0.00001)
        in_lat_max = st.sidebar.number_input("Max Latitude", value=max_lat_val, format="%.6f", step=0.00001)

        st.sidebar.subheader("📍 Precise Longitude")
        min_lon_val = float(matrix_df['GPS_0020_Lon'].min())
        max_lon_val = float(matrix_df['GPS_0020_Lon'].max())
        in_lon_min = st.sidebar.number_input("Min Longitude", value=min_lon_val, format="%.6f", step=0.00001)
        in_lon_max = st.sidebar.number_input("Max Longitude", value=max_lon_val, format="%.6f", step=0.00001)

        st.sidebar.divider()

        # 3. Additional Advanced Filters
        st.sidebar.subheader("📏 Altitude & Speed")
        h_min = st.sidebar.number_input("Min Height (m)", value=0.0, step=0.1)
        h_max = st.sidebar.number_input("Max Height (m)", value=60.0, step=0.1)
        
        speed_col = 'GPS_0020_gSpeed'
        s_min, s_max = 0.0, 10.0
        if speed_col in matrix_df.columns:
            s_min = st.sidebar.number_input("Min Speed (m/s)", value=0.0, step=0.1)
            s_max = st.sidebar.number_input("Max Speed (m/s)", value=float(matrix_df[speed_col].max()), step=0.1)

        st.sidebar.subheader("✨ Noise Filtering")
        p_range = st.sidebar.slider("Concentration Percentile (Remove Spikes)", 0, 100, (0, 100))

        # --- APPLY ALL FILTERS ---
        df_f = matrix_df.copy()
        df_f = df_f[df_f['GPS_0020_Lat'].between(in_lat_min, in_lat_max)]
        df_f = df_f[df_f['GPS_0020_Lon'].between(in_lon_min, in_lon_max)]
        df_f = df_f[df_f['GPS_0020_Height'].between(h_min, h_max)]
        
        if speed_col in df_f.columns:
            df_f = df_f[df_f[speed_col].between(s_min, s_max)]

        if not df_f.empty:
            low_p = np.percentile(df_f[sel_sub].dropna(), p_range[0])
            high_p = np.percentile(df_f[sel_sub].dropna(), p_range[1])
            df_f = df_f[df_f[sel_sub].between(low_p, high_p)]

        plot_df = df_f.dropna(subset=['GPS_0020_Lat', 'GPS_0020_Lon', sel_sub])

        # --- DASHBOARD ---
        st.title("⚰️ Grave Detection & Analysis Dashboard")
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Points Found", len(plot_df))
        m2.metric("Avg Conc", f"{plot_df[sel_sub].mean():.3f}" if not plot_df.empty else "0")
        m3.metric("Max Conc", f"{plot_df[sel_sub].max():.3f}" if not plot_df.empty else "0")

        t1, t2, t3 = st.tabs(["🗺️ Satellite Map", "📈 3D View", "📋 Raw Data"])

        with t1:
            if not plot_df.empty:
                fig_map = px.scatter_map(
                    plot_df, lat='GPS_0020_Lat', lon='GPS_0020_Lon',
                    color=sel_sub, size_max=12, zoom=19,
                    color_continuous_scale="Viridis", height=750
                )
                fig_map.update_layout(
                    map_style="white-bg",
                    map_layers=[{
                        "below": 'traces', "sourcetype": "raster",
                        "source": ["https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"]
                    }]
                )
                st.plotly_chart(fig_map, use_container_width=True)
            else:
                st.info("No points match the selected filters.")

        with t2:
            if not plot_df.empty:
                fig_3d = px.scatter_3d(plot_df, x='GPS_0020_Lon', y='GPS_0020_Lat', z=sel_sub, color=sel_sub, height=750)
                st.plotly_chart(fig_3d, use_container_width=True)

        with t3:
            st.dataframe(plot_df, use_container_width=True)
            st.download_button("Export as CSV", plot_df.to_csv(index=False), "survey_data.csv")

else:
    st.info("👋 Please upload your .json file in the sidebar to begin high-precision analysis.")
