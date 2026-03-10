import streamlit as st
import json
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import griddata
from streamlit_plotly_events import plotly_events
import folium
from streamlit_folium import st_folium
from folium.plugins import Draw

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Forensic Survey Dashboard", layout="wide")

# --- DATA LOADER ---
@st.cache_data
def load_and_process_data(uploaded_file):
    all_data = []
    uploaded_file.seek(0)
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
    df['Time_Sec'] = (df['vT']/1000).round().astype(int)
    # Ensure unique coordinates/time for interpolation
    df = df.groupby('Time_Sec').first().reset_index()
    return df

# --- SIDEBAR ---
st.sidebar.title("Survey Controls")
project_name = st.sidebar.text_input("Project Name", "Forensic Survey")
uploaded_file = st.sidebar.file_uploader("Upload JSON", type=["json"])

if uploaded_file:
    matrix_df = load_and_process_data(uploaded_file)
    if matrix_df is None:
        st.error("No valid data found")
        st.stop()
    st.sidebar.success("Data loaded")

    # --- SELECT SUBSTANCE ---
    cols = [c for c in matrix_df.columns if "STABSPECTRO" in c]
    if not cols:
        st.error("No chemical sensor data (STABSPECTRO) found in file.")
        st.stop()
    sel_sub = st.sidebar.selectbox("Substance", cols)

    # --- AREA FILTER ---
    st.sidebar.subheader("Area Filter")
    min_lat, max_lat = float(matrix_df["GPS_0020_Lat"].min()), float(matrix_df["GPS_0020_Lat"].max())
    min_lon, max_lon = float(matrix_df["GPS_0020_Lon"].min()), float(matrix_df["GPS_0020_Lon"].max())

    lat_min = st.sidebar.number_input("Min Latitude", value=min_lat, format="%.6f")
    lat_max = st.sidebar.number_input("Max Latitude", value=max_lat, format="%.6f")
    lon_min = st.sidebar.number_input("Min Longitude", value=min_lon, format="%.6f")
    lon_max = st.sidebar.number_input("Max Longitude", value=max_lon_val if 'max_lon_val' in locals() else max_lon, format="%.6f")

    # --- INITIAL FILTERING ---
    df_filtered = matrix_df.copy()
    df_filtered = df_filtered[df_filtered["GPS_0020_Lat"].between(lat_min, lat_max)]
    df_filtered = df_filtered[df_filtered["GPS_0020_Lon"].between(lon_min, lon_max)]
    plot_df = df_filtered.dropna(subset=["GPS_0020_Lat", "GPS_0020_Lon", sel_sub])

    # --- SESSION DATASET ---
    if "active_df" not in st.session_state or st.sidebar.button("Reset Dataset"):
        st.session_state.active_df = plot_df.copy()

    data = st.session_state.active_df.copy()

    # --- AI CONTROLS ---
    st.sidebar.subheader("AI Analysis")
    run_cluster = st.sidebar.checkbox("Cluster Detection")
    run_anomaly = st.sidebar.checkbox("Anomaly Detection")

    if run_cluster and len(data) > 10:
        coords = data[["GPS_0020_Lat", "GPS_0020_Lon"]]
        X = StandardScaler().fit_transform(coords)
        db = DBSCAN(eps=0.3, min_samples=5).fit(X)
        data["cluster"] = db.labels_
    else:
        data["cluster"] = -1

    if run_anomaly and len(data) > 1:
        std = data[sel_sub].std()
        if std > 0:
            z = np.abs((data[sel_sub] - data[sel_sub].mean()) / std)
            data["anomaly"] = z > 2.5
        else:
            data["anomaly"] = False
    else:
        data["anomaly"] = False

    # --- METRICS ---
    st.title(project_name)
    m1, m2, m3 = st.columns(3)
    m1.metric("Points", len(data))
    m2.metric("Average", round(data[sel_sub].mean(), 3))
    m3.metric("Max", round(data[sel_sub].max(), 3))

    # --- TABS ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Satellite Map", "Polygon Tool", "Heatmap", "Contour Map", "3D View"
    ])

    with tab1:
        fig = px.scatter_map(
            data, lat="GPS_0020_Lat", lon="GPS_0020_Lon", color=sel_sub,
            zoom=19, height=750, color_continuous_scale="Viridis"
        )
        fig.update_layout(
            dragmode="lasso", map_style="white-bg",
            map_layers=[{"below": "traces", "sourcetype": "raster",
                         "source": ["https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"]}]
        )
        # lasso capture
        selected = plotly_events(fig, select_event=True, override_height=750)

        if selected:
            st.warning(f"{len(selected)} points selected")
            if st.button("Remove Selected Points"):
                idx_to_drop = [p["pointIndex"] for p in selected]
                st.session_state.active_df = st.session_state.active_df.drop(
                    st.session_state.active_df.index[idx_to_drop]
                ).reset_index(drop=True)
                st.rerun()

    with tab2:
        st.subheader("Exclusion Polygon Tool")
        center = [data["GPS_0020_Lat"].mean(), data["GPS_0020_Lon"].mean()]
        m = folium.Map(location=center, zoom_start=19)
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri', name='Satellite', overlay=False, control=True
        ).add_to(m)
        Draw(export=True).add_to(m)
        map_data = st_folium(m, width=900, height=600)
        st.info("Use the draw tools on the left to mark areas. Note: Integration with the dataset requires geo-spatial intersection logic.")

    with tab3:
        heat = px.density_map(
            data, lat="GPS_0020_Lat", lon="GPS_0020_Lon", z=sel_sub,
            radius=15, zoom=18, height=750
        )
        heat.update_layout(
            map_style="white-bg",
            map_layers=[{"below": "traces", "sourcetype": "raster",
                         "source": ["https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"]}]
        )
        st.plotly_chart(heat, use_container_width=True)

    with tab4:
        st.subheader("Interpolated Contour Map")
        if len(data) > 3:
            grid_x, grid_y = np.mgrid[
                data["GPS_0020_Lon"].min():data["GPS_0020_Lon"].max():200j,
                data["GPS_0020_Lat"].min():data["GPS_0020_Lat"].max():200j
            ]
            grid_z = griddata(
                (data["GPS_0020_Lon"], data["GPS_0020_Lat"]),
                data[sel_sub],
                (grid_x, grid_y),
                method="linear"
            )
            # Display using imshow with coordinates
            fig_contour = px.imshow(
                grid_z.T, origin="lower", 
                x=np.linspace(data["GPS_0020_Lon"].min(), data["GPS_0020_Lon"].max(), 200),
                y=np.linspace(data["GPS_0020_Lat"].min(), data["GPS_0020_Lat"].max(), 200),
                labels=dict(x="Longitude", y="Latitude", color=sel_sub),
                color_continuous_scale="Viridis"
            )
            st.plotly_chart(fig_contour, use_container_width=True)
        else:
            st.info("Not enough data points for interpolation.")

    with tab5:
        fig3d = px.scatter_3d(
            data, x="GPS_0020_Lon", y="GPS_0020_Lat", z=sel_sub,
            color=sel_sub, height=750, title="Chemical Concentration 3D Profile"
        )
        st.plotly_chart(fig3d, use_container_width=True)

    # --- EXPORT ---
    st.sidebar.subheader("Export")
    csv = data.to_csv(index=False).encode("utf-8")
    st.sidebar.download_button(
        label="Download Cleaned CSV",
        data=csv,
        file_name=f"{project_name.lower().replace(' ', '_')}_cleaned.csv",
        mime="text/csv"
    )

else:
    st.info("Upload JSON survey file to begin.")
