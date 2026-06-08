import streamlit as st
import json
import pandas as pd
import plotly.express as px
import numpy as np
import os
from sklearn.cluster import DBSCAN
import math

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Grave Analysis Dashboard", layout="wide")

@st.cache_data
def load_and_process_data(uploaded_file):
    all_data = []
    # Reset file pointer to the beginning
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
    # Ensure unique rows per second
    return df.groupby('Time_Sec').first().reset_index()

# --- SIDEBAR GUI ---
st.sidebar.title("🛠️ Precise Control Panel")

# 1. Project Management
project_name = st.sidebar.text_input("Project Name", value="New Survey Site")

uploaded_file = st.sidebar.file_uploader("Upload .json data file", type=["json"])

if uploaded_file is not None:
    matrix_df = load_and_process_data(uploaded_file)
    
    if matrix_df is not None:
        st.sidebar.success("Data Loaded Successfully!")
        
        # 2. Isotope/Substance Selection
        cols = [c for c in matrix_df.columns if 'STABSPECTRO' in c and not c.startswith('s')]
        if not cols:
            # Fallback if specific strings aren't found, grab numeric columns
            cols = matrix_df.select_dtypes(include=[np.number]).columns.tolist()
        
        sel_sub = st.sidebar.selectbox("Analyze Substance", cols)

        st.sidebar.divider()

        # 3. High-Precision Numerical Inputs (6 Decimals for sub-meter accuracy)
        st.sidebar.subheader("📍 Area of Interest")
        min_lat_val = float(matrix_df['GPS_0020_Lat'].min()) if 'GPS_0020_Lat' in matrix_df else 0.0
        max_lat_val = float(matrix_df['GPS_0020_Lat'].max()) if 'GPS_0020_Lat' in matrix_df else 0.0
        in_lat_min = st.sidebar.number_input("Min Latitude", value=min_lat_val, format="%.6f", step=0.00001)
        in_lat_max = st.sidebar.number_input("Max Latitude", value=max_lat_val, format="%.6f", step=0.00001)

        min_lon_val = float(matrix_df['GPS_0020_Lon'].min()) if 'GPS_0020_Lon' in matrix_df else 0.0
        max_lon_val = float(matrix_df['GPS_0020_Lon'].max()) if 'GPS_0020_Lon' in matrix_df else 0.0
        in_lon_min = st.sidebar.number_input("Min Longitude", value=min_lon_val, format="%.6f", step=0.00001)
        in_lon_max = st.sidebar.number_input("Max Longitude", value=max_lon_val, format="%.6f", step=0.00001)

        st.sidebar.divider()

        # 4. Advanced Operational Filters
        st.sidebar.subheader("📏 Altitude & Speed")
        h_min = st.sidebar.number_input("Min Height (m)", value=0.0, step=0.1)
        h_max = st.sidebar.number_input("Max Height (m)", value=60.0, step=0.1)
        
        speed_col = 'GPS_0020_gSpeed'
        s_min, s_max = 0.0, 10.0
        if speed_col in matrix_df.columns:
            s_min = st.sidebar.number_input("Min Walking Speed (m/s)", value=0.0, step=0.1)
            s_max = st.sidebar.number_input("Max Walking Speed (m/s)", value=float(matrix_df[speed_col].max()), step=0.1)

        st.sidebar.subheader("✨ Noise Reduction")
        p_range = st.sidebar.slider("Concentration Percentile (Remove Outliers)", 0, 100, (0, 100))

        # --- APPLY ALL FILTERS ---
        df_f = matrix_df.copy()
        if 'GPS_0020_Lat' in df_f.columns and 'GPS_0020_Lon' in df_f.columns:
            df_f = df_f[df_f['GPS_0020_Lat'].between(in_lat_min, in_lat_max)]
            df_f = df_f[df_f['GPS_0020_Lon'].between(in_lon_min, in_lon_max)]
        
        if 'GPS_0020_Height' in df_f.columns:
            df_f = df_f[df_f['GPS_0020_Height'].between(h_min, h_max)]
        
        if speed_col in df_f.columns:
            df_f = df_f[df_f[speed_col].between(s_min, s_max)]

        if not df_f.empty and sel_sub in df_f.columns:
            low_p = np.percentile(df_f[sel_sub].dropna(), p_range[0])
            high_p = np.percentile(df_f[sel_sub].dropna(), p_range[1])
            df_f = df_f[df_f[sel_sub].between(low_p, high_p)]

        # Final filtered dataframe for visualization
        plot_df = df_f.dropna(subset=['GPS_0020_Lat', 'GPS_0020_Lon', sel_sub])

        # --- SIDEBAR EXPORT ---
        st.sidebar.divider()
        st.sidebar.subheader("📥 Export Results")
        safe_fn = "".join([c for c in project_name if c.isalnum() or c in (' ', '_')]).strip().replace(' ', '_')
        csv_data = plot_df.to_csv(index=False).encode('utf-8')
        st.sidebar.download_button(
            label="Download Filtered CSV",
            data=csv_data,
            file_name=f"{safe_fn}_filtered_data.csv",
            mime="text/csv"
        )

        # --- MAIN DASHBOARD DISPLAY ---
        st.title(f"⚰️ {project_name}")
        
        # Key Performance Indicators
        m1, m2, m3 = st.columns(3)
        m1.metric("Points Found", len(plot_df))
        m2.metric("Avg Concentration", f"{plot_df[sel_sub].mean():.3f}" if not plot_df.empty else "0")
        m3.metric("Max Concentration", f"{plot_df[sel_sub].max():.3f}" if not plot_df.empty else "0")

        # Tabs for different perspectives (Added Smart Detection)
        t1, t2, t3, t4 = st.tabs(["🗺️ Satellite Map", "📈 3D View", "📋 Raw Data", "🔍 Smart Detection"])

        with t1:
            if not plot_df.empty:
                # High-precision scatter plot on top of Esri Satellite imagery
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
                st.info("No data points match your current filters. Adjust your coordinates or height.")

        with t2:
            if not plot_df.empty:
                # 3D representation to visualize spikes in concentration
                fig_3d = px.scatter_3d(
                    plot_df, x='GPS_0020_Lon', y='GPS_0020_Lat', z=sel_sub, 
                    color=sel_sub, height=750, color_continuous_scale="Viridis"
                )
                st.plotly_chart(fig_3d, use_container_width=True)

        with t3:
            st.subheader(f"Raw Matrix: {project_name}")
            st.dataframe(plot_df, use_container_width=True)

        with t4:
            st.subheader("Automated Hotspot & Grave Detection")
            st.markdown("This tool uses **DBSCAN spatial clustering** to isolate anomalies. It filters for high concentrations of the selected substance and groups contiguous spikes into predicted grave locations.")
            
            c1, c2 = st.columns(2)
            with c1:
                anomaly_thresh = st.slider("Anomaly Threshold (Percentile)", min_value=50, max_value=99, value=90, 
                                           help="Only consider data points in the top X percentile of concentration as potential indicators.")
            with c2:
                eps_meters = st.slider("Cluster Radius (meters)", min_value=0.5, max_value=5.0, value=1.5, step=0.1,
                                       help="Maximum distance between two points for them to be considered part of the same grave.")

            if not plot_df.empty:
                # 1. Filter for anomalies based on user threshold
                threshold_val = np.percentile(plot_df[sel_sub], anomaly_thresh)
                anomalies_df = plot_df[plot_df[sel_sub] >= threshold_val].copy()

                if not anomalies_df.empty and len(anomalies_df) >= 3:
                    # 2. Convert meters to approximate degrees for clustering
                    # 1 degree of latitude is approx 111,320 meters
                    eps_degrees = eps_meters / 111320.0
                    
                    coords = anomalies_df[['GPS_0020_Lat', 'GPS_0020_Lon']].values
                    
                    # 3. Run DBSCAN Clustering
                    # min_samples=3 requires at least 3 high-concentration pings to validate a cluster (filters out single-ping noise)
                    db = DBSCAN(eps=eps_degrees, min_samples=3).fit(coords)
                    anomalies_df['Cluster_ID'] = db.labels_
                    
                    # -1 means noise (unclustered) in DBSCAN
                    valid_clusters = anomalies_df[anomalies_df['Cluster_ID'] != -1]
                    
                    if not valid_clusters.empty:
                        # 4. Calculate Centroids for the predicted graves
                        centroids = valid_clusters.groupby('Cluster_ID').agg({
                            'GPS_0020_Lat': 'mean',
                            'GPS_0020_Lon': 'mean',
                            sel_sub: 'max' # Record the peak concentration for this grave
                        }).reset_index()
                        
                        st.success(f"**{len(centroids)} potential grave locations detected!**")
                        
                        # Display Map with both anomalies and centroids
                        fig_smart = px.scatter_map(
                            valid_clusters, lat='GPS_0020_Lat', lon='GPS_0020_Lon',
                            color='Cluster_ID', zoom=19, height=600,
                            title="Detected Clusters (Points)",
                            color_discrete_sequence=px.colors.qualitative.Pastel
                        )
                        
                        # Add large distinct markers for the exact Centroid (Grave Center)
                        fig_smart.add_scattermap(
                            lat=centroids['GPS_0020_Lat'],
                            lon=centroids['GPS_0020_Lon'],
                            mode='markers+text',
                            marker=dict(size=15, color='red', symbol='circle'),
                            text=[f"Grave {i+1}" for i in range(len(centroids))],
                            textposition="top right",
                            name="Predicted Centers"
                        )

                        fig_smart.update_layout(
                            map_style="white-bg",
                            map_layers=[{
                                "below": 'traces', "sourcetype": "raster",
                                "source": ["https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"]
                            }],
                            showlegend=False
                        )
                        st.plotly_chart(fig_smart, use_container_width=True)
                        
                        # Dataframe of suspected coordinates
                        st.write("### Predicted Center Coordinates")
                        st.dataframe(centroids.rename(columns={
                            'GPS_0020_Lat': 'Latitude', 
                            'GPS_0020_Lon': 'Longitude', 
                            sel_sub: 'Peak Concentration'
                        }).drop(columns=['Cluster_ID']), use_container_width=True)

                    else:
                        st.warning("Anomalies were found, but they were too scattered to form distinct gravesite clusters. Try increasing the Cluster Radius or lowering the Anomaly Threshold.")
                else:
                    st.warning("Not enough high-concentration points found to cluster. Try lowering the Anomaly Threshold.")
            else:
                st.info("No data available to run detection.")

else:
    st.info("👋 Welcome! Please upload your .json survey file in the sidebar to begin analysis.")
