import streamlit as st
import json
import pandas as pd
import plotly.express as px
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Sensor Data Analysis", layout="wide")

# --- DATA PROCESSING FUNCTIONS ---
@st.cache_data
def load_and_process_data(uploaded_file):
    all_data = []
    
    # Read the uploaded file line by line
    for line in uploaded_file:
        line = line.decode("utf-8").strip()
        if not line:
            continue
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
        except:
            continue
    
    if not all_data:
        return None
    
    df = pd.DataFrame(all_data)
    df['Time_Sec'] = (df['vT'] / 1000).round().astype(int)
    # Grouping to ensure one row per second
    matrix = df.groupby('Time_Sec').first().reset_index()
    return matrix

# --- SIDEBAR GUI ---
st.sidebar.title("🛠️ Control Panel")

# GitHub/Cloud versions must use a file uploader instead of a local C:\ path
uploaded_file = st.sidebar.file_uploader("Upload your .json data file", type=["json"])

if uploaded_file is not None:
    matrix_df = load_and_process_data(uploaded_file)
    
    if matrix_df is not None:
        st.sidebar.success("Data Loaded!")
        
        # 1. Substance Selection
        cols = [c for c in matrix_df.columns if 'STABSPECTRO' in c and not c.startswith('s')]
        sel_sub = st.sidebar.selectbox("Analyze Substance", cols)

        # 2. Filtering Controls
        st.sidebar.subheader("Filters")
        lat_range = st.sidebar.slider("Lat Range", 50.0, 54.0, (50.750, 53.555), step=0.001)
        lon_range = st.sidebar.slider("Lon Range", 3.0, 8.0, (3.358, 7.228), step=0.001)
        
        use_h = st.sidebar.checkbox("Filter Height", value=True)
        h_range = st.sidebar.slider("Height (m)", 0, 100, (0, 55))

        # --- APPLY FILTERS ---
        df_f = matrix_df.copy()
        df_f = df_f[df_f['GPS_0020_Lat'].between(lat_range[0], lat_range[1])]
        df_f = df_f[df_f['GPS_0020_Lon'].between(lon_range[0], lon_range[1])]
        
        if use_h and 'GPS_0020_Height' in df_f.columns:
            df_f = df_f[df_f['GPS_0020_Height'].between(h_range[0], h_range[1])]

        # Final cleaning
        plot_df = df_f.dropna(subset=['GPS_0020_Lat', 'GPS_0020_Lon', sel_sub])

        # --- MAIN DASHBOARD ---
        st.title("☢️ Radiation Survey Dashboard")
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Points", len(plot_df))
        m2.metric("Mean Conc", round(plot_df[sel_sub].mean(), 2) if not plot_df.empty else 0)
        m3.metric("Max Conc", round(plot_df[sel_sub].max(), 2) if not plot_df.empty else 0)

        t1, t2, t3 = st.tabs(["🗺️ Map", "📈 3D View", "📋 Table"])

        with t1:
            if not plot_df.empty:
                fig_map = px.scatter_map(
                    plot_df, lat='GPS_0020_Lat', lon='GPS_0020_Lon',
                    color=sel_sub, size_max=15, zoom=14,
                    color_continuous_scale="Viridis", height=600
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
                st.info("No data points match your filters.")

        with t2:
            if not plot_df.empty:
                fig_3d = px.scatter_3d(
                    plot_df, x='GPS_0020_Lon', y='GPS_0020_Lat', z=sel_sub,
                    color=sel_sub, color_continuous_scale="Viridis", height=600
                )
                st.plotly_chart(fig_3d, use_container_width=True)

        with t3:
            st.dataframe(plot_df)
            st.download_button("Export CSV", plot_df.to_csv(index=False), "data.csv")
    else:
        st.error("The uploaded file could not be parsed. Please check the format.")
else:
    st.info("👋 Please upload a .json file in the sidebar to begin.")
