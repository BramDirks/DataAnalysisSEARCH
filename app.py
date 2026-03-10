import streamlit as st
import json
import pandas as pd
import plotly.express as px
import numpy as np
import os
from streamlit_plotly_events import plotly_events

# --- PANDAS CONFIGURATION ---
# This fixes the "max_elements" error you encountered
pd.set_option("styler.render.max_elements", 500000) 

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="High-Precision Grave Analysis", layout="wide")

# Initialize Session State for exclusions
if 'excluded_indices' not in st.session_state:
    st.session_state.excluded_indices = set()

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

# --- SIDEBAR GUI ---
st.sidebar.title("🛠️ Precise Control Panel")

# Project Name for Export
project_name = st.sidebar.text_input("Project Name", value="Grave_Site_Alpha")

uploaded_file = st.sidebar.file_uploader("Upload .json data file", type=["json"])

if st.sidebar.button("🔄 Reset Manual Exclusions"):
    st.session_state.excluded_indices = set()
    st.rerun()

if uploaded_file is not None:
    matrix_df = load_and_process_data(uploaded_file)
    
    if matrix_df is not None:
        st.sidebar.success("Data Loaded!")
        
        cols = [c for c in matrix_df.columns if 'STABSPECTRO' in c and not c.startswith('s')]
        sel_sub = st.sidebar.selectbox("Analyze Substance", cols)

        st.sidebar.subheader("📍 Precise Coordinates")
        in_lat_min = st.sidebar.number_input("Min Lat", value=float(matrix_df['GPS_0020_Lat'].min()), format="%.6f")
        in_lat_max = st.sidebar.number_input("Max Lat", value=float(matrix_df['GPS_0020_Lat'].max()), format="%.6f")
        in_lon_min = st.sidebar.number_input("Min Lon", value=float(matrix_df['GPS_0020_Lon'].min()), format="%.6f")
        in_lon_max = st.sidebar.number_input("Max Lon", value=float(matrix_df['GPS_0020_Lon'].max()), format="%.6f")

        # --- APPLY FILTERS ---
        df_f = matrix_df.copy()
        df_f = df_f[df_f['GPS_0020_Lat'].between(in_lat_min, in_lat_max)]
        df_f = df_f[df_f['GPS_0020_Lon'].between(in_lon_min, in_lon_max)]
        
        df_f['is_excluded'] = df_f['point_id'].isin(st.session_state.excluded_indices)
        plot_df = df_f[~df_f['is_excluded']].dropna(subset=[sel_sub])

        # --- MAIN DASHBOARD ---
        st.title(f"⚰️ Analysis: {project_name}")
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Active Points", len(plot_df))
        m2.metric("Mean Conc", f"{plot_df[sel_sub].mean():.3f}" if not plot_df.empty else "0")
        m3.metric("Excluded", len(st.session_state.excluded_indices))

        t1, t2, t3 = st.tabs(["🗺️ Exclusion Map", "📈 3D View", "📋 Data Table"])

        with t1:
            st.info("💡 Use Box/Lasso select on the map to exclude points. They will turn grey.")
            fig_map = px.scatter_map(
                df_f, lat='GPS_0020_Lat', lon='GPS_0020_Lon',
                color=sel_sub, 
                opacity=df_f['is_excluded'].map({True: 0.15, False: 0.9}),
                size_max=12, zoom=19,
                color_continuous_scale="Viridis", height=700,
                hover_data=['point_id']
            )
            fig_map.update_layout(
                map_style="white-bg",
                map_layers=[{"below": 'traces', "sourcetype": "raster", "source": ["https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"]}],
                clickmode='event+select'
            )
            
            selected_points = plotly_events(fig_map, select_event=True, key="map_selection")
            if selected_points:
                new_ids = [df_f.iloc[p['pointIndex']]['point_id'] for p in selected_points]
                st.session_state.excluded_indices.update(new_ids)
                st.rerun()

        with t2:
            if not plot_df.empty:
                fig_3d = px.scatter_3d(plot_df, x='GPS_0020_Lon', y='GPS_0020_Lat', z=sel_sub, color=sel_sub, height=750)
                st.plotly_chart(fig_3d, use_container_width=True)

        with t3:
            st.subheader(f"Table View for {project_name}")
            
            # Use styling for excluded points
            def style_rows(row):
                if row['is_excluded']:
                    return ['background-color: #ffcccc; color: red;'] * len(row)
                return [''] * len(row)

            # Render the styled table
            st.dataframe(
                df_f.style.apply(style_rows, axis=1),
                use_container_width=True
            )
            
            # Export CSV includes the project name in the filename
            filename = f"{project_name.replace(' ', '_')}_export.csv"
            st.download_button("Export Clean CSV", plot_df.to_csv(index=False), filename)

else:
    st.info("👋 Please upload your .json file.")
