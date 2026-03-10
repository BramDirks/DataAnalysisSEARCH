import streamlit as st
import json
import pandas as pd
import plotly.express as px
import numpy as np
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Grave Analysis Dashboard", layout="wide")

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
    return df.groupby('Time_Sec').first().reset_index()

# --- SIDEBAR GUI ---
st.sidebar.title("🛠️ Precise Control Panel")
project_name = st.sidebar.text_input("Project Name", value="New Survey Site")
uploaded_file = st.sidebar.file_uploader("Upload .json data file", type=["json"])

if uploaded_file is not None:
    matrix_df = load_and_process_data(uploaded_file)
    
    if matrix_df is not None:
        st.sidebar.success("Data Loaded Successfully!")
        
        # 1. MULTI-SUBSTANCE SELECTION
        cols = [c for c in matrix_df.columns if 'STABSPECTRO' in c and not c.startswith('s')]
        selected_subs = st.sidebar.multiselect("Select Substances to Layer", options=cols, default=[cols[0]])

        st.sidebar.divider()

        # 2. High-Precision Numerical Inputs
        st.sidebar.subheader("📍 Area of Interest")
        min_lat_val, max_lat_val = float(matrix_df['GPS_0020_Lat'].min()), float(matrix_df['GPS_0020_Lat'].max())
        in_lat_min = st.sidebar.number_input("Min Latitude", value=min_lat_val, format="%.6f", step=0.00001)
        in_lat_max = st.sidebar.number_input("Max Latitude", value=max_lat_val, format="%.6f", step=0.00001)

        min_lon_val, max_lon_val = float(matrix_df['GPS_0020_Lon'].min()), float(matrix_df['GPS_0020_Lon'].max())
        in_lon_min = st.sidebar.number_input("Min Longitude", value=min_lon_val, format="%.6f", step=0.00001)
        in_lon_max = st.sidebar.number_input("Max Longitude", value=max_lon_val, format="%.6f", step=0.00001)

        st.sidebar.divider()

        # 3. Filters
        st.sidebar.subheader("📏 Altitude & Speed")
        h_min = st.sidebar.number_input("Min Height (m)", value=0.0, step=0.1)
        h_max = st.sidebar.number_input("Max Height (m)", value=60.0, step=0.1)
        
        speed_col = 'GPS_0020_gSpeed'
        s_min = st.sidebar.number_input("Min Walking Speed (m/s)", value=0.0)

        st.sidebar.subheader("✨ Noise Filtering")
        p_range = st.sidebar.slider("Concentration Percentile", 0, 100, (0, 100))

        # --- APPLY GLOBAL FILTERS ---
        df_f = matrix_df.copy()
        df_f = df_f[df_f['GPS_0020_Lat'].between(in_lat_min, in_lat_max)]
        df_f = df_f[df_f['GPS_0020_Lon'].between(in_lon_min, in_lon_max)]
        df_f = df_f[df_f['GPS_0020_Height'].between(h_min, h_max)]
        if speed_col in df_f.columns:
            df_f = df_f[df_f[speed_col] >= s_min]

        # --- MAIN DASHBOARD DISPLAY ---
        st.title(f"⚰️ {project_name}")
        
        t1, t2, t3 = st.tabs(["🗺️ Multi-Layer Map", "📈 3D Profile", "📋 Data Matrix"])

        with t1:
            st.subheader("Overlapping Sensor Layers")
            if not df_f.empty and selected_subs:
                # We "melt" the data so Plotly can handle multiple columns as one legend
                melt_df = df_f.melt(id_vars=['GPS_0020_Lat', 'GPS_0020_Lon'], 
                                   value_vars=selected_subs, 
                                   var_name='Substance', 
                                   value_name='Concentration')
                
                # Filter noise for each substance individually within the melted set
                final_plot_df = []
                for sub in selected_subs:
                    sub_data = melt_df[melt_df['Substance'] == sub].dropna()
                    if not sub_data.empty:
                        low_val = np.percentile(sub_data['Concentration'], p_range[0])
                        high_val = np.percentile(sub_data['Concentration'], p_range[1])
                        final_plot_df.append(sub_data[sub_data['Concentration'].between(low_val, high_val)])
                
                if final_plot_df:
                    plot_df = pd.concat(final_plot_df)
                    
                    fig_map = px.scatter_map(
                        plot_df, 
                        lat='GPS_0020_Lat', 
                        lon='GPS_0020_Lon', 
                        color='Substance',        # Different color for each substance
                        size='Concentration',     # Size shows the intensity
                        size_max=15, 
                        zoom=19, 
                        height=750,
                        hover_name='Substance',
                        hover_data={'Concentration': ':.3f'}
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
                    st.info("No data points match the filtering criteria.")

        with t2:
            if not df_f.empty and selected_subs:
                # 3D plot using colors for different substances
                fig_3d = px.scatter_3d(
                    plot_df, x='GPS_0020_Lon', y='GPS_0020_Lat', z='Concentration',
                    color='Substance', height=800
                )
                st.plotly_chart(fig_3d, use_container_width=True)

        with t3:
            st.subheader("Raw Data Export")
            st.dataframe(df_f[['Time_Sec', 'GPS_0020_Lat', 'GPS_0020_Lon'] + selected_subs])
            csv_data = df_f.to_csv(index=False).encode('utf-8')
            st.download_button("Download Full Matrix CSV", data=csv_data, file_name=f"{project_name}_matrix.csv")

else:
    st.info("👋 Welcome! Please upload your .json survey file to begin.")
