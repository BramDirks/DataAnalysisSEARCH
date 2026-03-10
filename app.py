import streamlit as st
import json
import pandas as pd
import plotly.express as px
import numpy as np
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="High-Precision Sensor Analysis", layout="wide")

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
    # Time processing
    df['DateTime'] = pd.to_datetime(df['vT'], unit='ms')
    df['Time_Sec'] = (df['vT'] / 1000).round().astype(int)
    return df.groupby('Time_Sec').first().reset_index()

# --- SIDEBAR GUI ---
st.sidebar.title("🛠️ Precise Control Panel")

uploaded_file = st.sidebar.file_uploader("Upload .json data", type=["json"])

if uploaded_file is not None:
    matrix_df = load_and_process_data(uploaded_file)
    
    if matrix_df is not None:
        st.sidebar.success("Data Loaded!")
        
        # 1. SUBSTANCE SELECTION
        cols = [c for c in matrix_df.columns if 'STABSPECTRO' in c and not c.startswith('s')]
        sel_sub = st.sidebar.selectbox("Analyze Substance", cols)

        st.sidebar.divider()
        
        # 2. MANUAL COORDINATE INPUTS (Replacing Sliders)
        st.sidebar.subheader("📍 Precise Coordinates")
        c1, c2 = st.sidebar.columns(2)
        min_lat = c1.number_input("Min Lat", value=float(matrix_df['GPS_0020_Lat'].min()), format="%.6f")
        max_lat = c2.number_input("Max Lat", value=float(matrix_df['GPS_0020_Lat'].max()), format="%.6f")
        
        c3, c4 = st.sidebar.columns(2)
        min_lon = c3.number_input("Min Lon", value=float(matrix_df['GPS_0020_Lon'].min()), format="%.6f")
        max_lon = c4.number_input("Max Lon", value=float(matrix_df['GPS_0020_Lon'].max()), format="%.6f")

        st.sidebar.divider()

        # 3. ADVANCED FILTERS
        st.sidebar.subheader("📏 Altitude & Speed")
        c5, c6 = st.sidebar.columns(2)
        h_min = c5.number_input("Min Height (m)", value=0.0)
        h_max = c6.number_input("Max Height (m)", value=60.0)
        
        speed_col = 'GPS_0020_gSpeed'
        if speed_col in matrix_df.columns:
            s_min = st.sidebar.number_input("Min Ground Speed (m/s)", value=0.0)
            s_max = st.sidebar.number_input("Max Ground Speed (m/s)", value=5.0)

        st.sidebar.subheader("✨ Data Cleaning")
        # Percentile filter to remove noise/outliers
        p_low, p_high = st.sidebar.select_slider(
            "Filter Concentration Percentile (Remove Outliers)",
            options=list(range(0, 101)),
            value=(0, 100)
        )

        # --- APPLY FILTERS ---
        df_f = matrix_df.copy()
        
        # Apply Logic
        df_f = df_f[df_f['GPS_0020_Lat'].between(min_lat, max_lat)]
        df_f = df_f[df_f['GPS_0020_Lon'].between(min_lon, max_lon)]
        df_f = df_f[df_f['GPS_0020_Height'].between(h_min, h_max)]
        
        if speed_col in df_f.columns:
            df_f = df_f[df_f[speed_col].between(s_min, s_max)]

        # Percentile Filter
        if not df_f.empty:
            low_val = np.percentile(df_f[sel_sub].dropna(), p_low)
            high_val = np.percentile(df_f[sel_sub].dropna(), p_high)
            df_f = df_f[df_f[sel_sub].between(low_val, high_val)]

        plot_df = df_f.dropna(subset=['GPS_0020_Lat', 'GPS_0020_Lon', sel_sub])

        # --- MAIN DASHBOARD ---
        st.title("⚰️ Archaeological Survey Dashboard")
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Points", len(plot_df))
        m2.metric("Mean Conc", round(plot_df[sel_sub].mean(), 3) if not plot_df.empty else 0)
        m3.metric("Max Conc", round(plot_df[sel_sub].max(), 3) if not plot_df.empty else 0)

        t1, t2, t3 = st.tabs(["🗺️ High-Res Map", "📈 3D Profile", "📋 Data Table"])

        with t1:
            if not plot_df.empty:
                # We use scatter_map for high precision
                fig_map = px.scatter_map(
                    plot_df, lat='GPS_0020_Lat', lon='GPS_0020_Lon',
                    color=sel_sub, size_max=12, zoom=18,
                    color_continuous_scale="Viridis", height=700
                )
                fig_map.update_layout(
                    map_style="white-bg",
                    map_layers=[{
                        "below": 'traces', "sourcetype": "raster",
                        "source": ["https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"]
                    }]
                )
                st.plotly_chart(fig_map, use_container_width=True)

        with t2:
            if not plot_df.empty:
                fig_3d = px.scatter_3d(
                    plot_df, x='GPS_0020_Lon', y='GPS_0020_Lat', z=sel_sub,
                    color=sel_sub, color_continuous_scale="Viridis", height=700
                )
                st.plotly_chart(fig_3d, use_container_width=True)

        with t3:
            st.dataframe(plot_df)
            st.download_button("Export Processed CSV", plot_df.to_csv(index=False), "survey_data.csv")
    else:
        st.error("The uploaded file could not be parsed.")
else:
    st.info("👋 Upload a .json file to begin. Use the number boxes for sub-meter precision.")
