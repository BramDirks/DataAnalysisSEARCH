import streamlit as st
import json
import pandas as pd
import plotly.express as px
import numpy as np
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Multi-Isotope Grave Analysis", layout="wide")

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
st.sidebar.title("🛠️ Multi-Sensor Control")
uploaded_file = st.sidebar.file_uploader("Upload .json data file", type=["json"])

if uploaded_file is not None:
    matrix_df = load_and_process_data(uploaded_file)
    
    if matrix_df is not None:
        st.sidebar.success("Data Loaded!")
        
        # 1. MULTI-SUBSTANCE SELECTION (Acts as the 'plus' and 'toggle')
        all_cols = [c for c in matrix_df.columns if 'STABSPECTRO' in c and not c.startswith('s')]
        selected_substances = st.sidebar.multiselect(
            "Select Substances to Compare", 
            options=all_cols, 
            default=[all_cols[0]] if all_cols else None
        )

        st.sidebar.divider()

        # 2. HIGH-PRECISION COORDINATES
        st.sidebar.subheader("📍 Precise Area (6 Decimals)")
        c1, c2 = st.sidebar.columns(2)
        in_lat_min = c1.number_input("Min Lat", value=float(matrix_df['GPS_0020_Lat'].min()), format="%.6f")
        in_lat_max = c2.number_input("Max Lat", value=float(matrix_df['GPS_0020_Lat'].max()), format="%.6f")

        c3, c4 = st.sidebar.columns(2)
        in_lon_min = c3.number_input("Min Lon", value=float(matrix_df['GPS_0020_Lon'].min()), format="%.6f")
        in_lon_max = c4.number_input("Max Lon", value=float(matrix_df['GPS_0020_Lon'].max()), format="%.6f")

        st.sidebar.divider()

        # 3. ADVANCED FILTERS
        st.sidebar.subheader("📏 Altitude & Speed")
        h_min = st.sidebar.number_input("Min Height (m)", value=0.0, step=0.1)
        h_max = st.sidebar.number_input("Max Height (m)", value=60.0, step=0.1)
        
        speed_col = 'GPS_0020_gSpeed'
        s_min, s_max = 0.0, 10.0
        if speed_col in matrix_df.columns:
            s_min = st.sidebar.number_input("Min Speed (m/s)", value=0.0, step=0.1)
            s_max = st.sidebar.number_input("Max Speed (m/s)", value=float(matrix_df[speed_col].max()), step=0.1)

        # --- APPLY ALL FILTERS ---
        df_f = matrix_df.copy()
        df_f = df_f[df_f['GPS_0020_Lat'].between(in_lat_min, in_lat_max)]
        df_f = df_f[df_f['GPS_0020_Lon'].between(in_lon_min, in_lon_max)]
        df_f = df_f[df_f['GPS_0020_Height'].between(h_min, h_max)]
        
        if speed_col in df_f.columns:
            df_f = df_f[df_f[speed_col].between(s_min, s_max)]

        # --- DASHBOARD ---
        st.title("⚰️ Multi-Isotope Archaeological Analysis")
        
        if not selected_substances:
            st.warning("Please select at least one substance in the sidebar.")
        else:
            # Stats for all selected substances
            cols = st.columns(len(selected_substances))
            for i, sub in enumerate(selected_substances):
                cols[i].metric(sub.split('_')[-1], f"{df_f[sub].max():.2f} (Max)")

            t1, t2, t3 = st.tabs(["🗺️ Comparative Map", "📈 3D Multi-Plot", "📋 Export Data"])

            with t1:
                # To show multiple substances on one map, we "melt" the dataframe
                map_df = df_f.melt(id_vars=['GPS_0020_Lat', 'GPS_0020_Lon'], 
                                   value_vars=selected_substances,
                                   var_name='Substance', value_name='Concentration')
                
                fig_map = px.scatter_map(
                    map_df, lat='GPS_0020_Lat', lon='GPS_0020_Lon',
                    color='Concentration', facet_col='Substance', # Creates side-by-side maps for comparison
                    size_max=12, zoom=18,
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

            with t2:
                # 3D Plot showing multiple isotopes as different colors/axes
                fig_3d = px.scatter_3d(
                    map_df, x='GPS_0020_Lon', y='GPS_0020_Lat', z='Concentration',
                    color='Substance', height=800, title="Multi-Substance Elevation Profile"
                )
                st.plotly_chart(fig_3d, use_container_width=True)

            with t3:
                st.dataframe(df_f[['DateTime', 'GPS_0020_Lat', 'GPS_0020_Lon'] + selected_substances])
                st.download_button("Download CSV", df_f.to_csv(index=False), "multi_isotope_data.csv")

else:
    st.info("👋 Upload a .json file to begin. You can now select and toggle multiple isotopes in the sidebar.")
