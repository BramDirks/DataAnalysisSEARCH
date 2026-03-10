import streamlit as st
import json
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from streamlit_plotly_events import plotly_events

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Forensic Survey Dashboard", layout="wide")

# -----------------------------
# DATA LOADER
# -----------------------------
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

    df['Time_Sec'] = (df['vT'] / 1000).round().astype(int)

    df = df.groupby('Time_Sec').first().reset_index()

    return df


# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("Survey Controls")

project_name = st.sidebar.text_input("Project Name", "Forensic Survey")

uploaded_file = st.sidebar.file_uploader("Upload JSON Data", type=["json"])

if uploaded_file:

    matrix_df = load_and_process_data(uploaded_file)

    if matrix_df is None:
        st.error("No usable data found")
        st.stop()

    st.sidebar.success("Data Loaded")

# -----------------------------
# SUBSTANCE SELECTION
# -----------------------------
    cols = [c for c in matrix_df.columns if 'STABSPECTRO' in c]

    sel_sub = st.sidebar.selectbox("Substance", cols)

# -----------------------------
# AREA FILTER
# -----------------------------
    st.sidebar.subheader("Area Filter")

    min_lat = float(matrix_df['GPS_0020_Lat'].min())
    max_lat = float(matrix_df['GPS_0020_Lat'].max())

    min_lon = float(matrix_df['GPS_0020_Lon'].min())
    max_lon = float(matrix_df['GPS_0020_Lon'].max())

    lat_min = st.sidebar.number_input("Min Latitude", value=min_lat, format="%.6f")
    lat_max = st.sidebar.number_input("Max Latitude", value=max_lat, format="%.6f")

    lon_min = st.sidebar.number_input("Min Longitude", value=min_lon, format="%.6f")
    lon_max = st.sidebar.number_input("Max Longitude", value=max_lon, format="%.6f")

# -----------------------------
# DATA FILTER
# -----------------------------
    df = matrix_df.copy()

    df = df[df['GPS_0020_Lat'].between(lat_min, lat_max)]
    df = df[df['GPS_0020_Lon'].between(lon_min, lon_max)]

    plot_df = df.dropna(subset=['GPS_0020_Lat', 'GPS_0020_Lon', sel_sub])

# -----------------------------
# SESSION STATE DATASET
# -----------------------------
    if "active_df" not in st.session_state:
        st.session_state.active_df = plot_df.copy()

    if st.sidebar.button("Reset Dataset"):
        st.session_state.active_df = plot_df.copy()

    data = st.session_state.active_df.copy()

# -----------------------------
# AI CONTROLS
# -----------------------------
    st.sidebar.subheader("AI Detection")

    run_cluster = st.sidebar.checkbox("Cluster Detection")

    run_anomaly = st.sidebar.checkbox("Anomaly Detection")

# -----------------------------
# CLUSTER DETECTION
# -----------------------------
    if run_cluster and len(data) > 5:

        coords = data[['GPS_0020_Lat', 'GPS_0020_Lon']]

        scaler = StandardScaler()
        X = scaler.fit_transform(coords)

        db = DBSCAN(eps=0.3, min_samples=5).fit(X)

        data['cluster'] = db.labels_

    else:
        data['cluster'] = -1

# -----------------------------
# ANOMALY DETECTION
# -----------------------------
    if run_anomaly:

        z = np.abs((data[sel_sub] - data[sel_sub].mean()) / data[sel_sub].std())

        data['anomaly'] = z > 2.5

    else:
        data['anomaly'] = False

# -----------------------------
# METRICS
# -----------------------------
    st.title(project_name)

    m1, m2, m3 = st.columns(3)

    m1.metric("Points", len(data))
    m2.metric("Average", round(data[sel_sub].mean(), 3))
    m3.metric("Max", round(data[sel_sub].max(), 3))

# -----------------------------
# TABS
# -----------------------------
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Satellite Map", "Heatmap", "3D View", "Raw Data"]
    )

# -----------------------------
# SATELLITE MAP
# -----------------------------
    with tab1:

        fig = px.scatter_mapbox(
            data,
            lat="GPS_0020_Lat",
            lon="GPS_0020_Lon",
            color=sel_sub,
            hover_data=[sel_sub],
            zoom=19,
            height=750
        )

        fig.update_layout(
            mapbox_style="satellite-streets",
            dragmode="lasso"
        )

        selected_points = plotly_events(
            fig,
            select_event=True,
            override_height=750
        )

        st.plotly_chart(fig, use_container_width=True)

        if selected_points:

            st.warning(f"{len(selected_points)} points selected")

            if st.button("Remove Selected Points"):

                indices = [p["pointIndex"] for p in selected_points]

                st.session_state.active_df = st.session_state.active_df.drop(
                    st.session_state.active_df.index[indices]
                ).reset_index(drop=True)

                st.success("Points removed")

                st.rerun()

# -----------------------------
# HEATMAP
# -----------------------------
    with tab2:

        heatmap = px.density_mapbox(
            data,
            lat="GPS_0020_Lat",
            lon="GPS_0020_Lon",
            z=sel_sub,
            radius=15,
            zoom=18,
            height=750
        )

        heatmap.update_layout(mapbox_style="satellite-streets")

        st.plotly_chart(heatmap, use_container_width=True)

# -----------------------------
# 3D VIEW
# -----------------------------
    with tab3:

        fig3d = px.scatter_3d(
            data,
            x="GPS_0020_Lon",
            y="GPS_0020_Lat",
            z=sel_sub,
            color=sel_sub,
            height=750
        )

        st.plotly_chart(fig3d, use_container_width=True)

# -----------------------------
# RAW DATA
# -----------------------------
    with tab4:

        st.dataframe(data, use_container_width=True)

# -----------------------------
# EXPORT
# -----------------------------
    st.sidebar.subheader("Export")

    csv = data.to_csv(index=False).encode("utf-8")

    st.sidebar.download_button(
        "Download Cleaned CSV",
        csv,
        file_name="filtered_data.csv",
        mime="text/csv"
    )

else:

    st.info("Upload a JSON survey file to begin analysis.")
