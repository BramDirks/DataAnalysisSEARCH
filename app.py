import streamlit as st
import json
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import griddata
import folium
from streamlit_folium import st_folium
from folium.plugins import Draw

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
                        for k,v in val.items():
                            row[f"{sensor['eID']}_{k}"] = v
                    else:
                        row[f"{sensor['eID']}_val"] = val

                    all_data.append(row)

        except:
            continue

    if not all_data:
        return None

    df = pd.DataFrame(all_data)

    df["Time_Sec"] = (df["vT"]/1000).round().astype(int)

    df = df.groupby("Time_Sec").first().reset_index()

    return df


# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("Survey Controls")

project_name = st.sidebar.text_input("Project Name","Forensic Survey")

uploaded_file = st.sidebar.file_uploader("Upload JSON Data",type=["json"])

if uploaded_file:

    matrix_df = load_and_process_data(uploaded_file)

    if matrix_df is None:
        st.error("No valid data found")
        st.stop()

    st.sidebar.success("Data loaded")

# -----------------------------
# SUBSTANCE
# -----------------------------
    cols = [c for c in matrix_df.columns if "STABSPECTRO" in c]

    sel_sub = st.sidebar.selectbox("Substance",cols)

# -----------------------------
# AREA FILTER
# -----------------------------
    min_lat = float(matrix_df["GPS_0020_Lat"].min())
    max_lat = float(matrix_df["GPS_0020_Lat"].max())

    min_lon = float(matrix_df["GPS_0020_Lon"].min())
    max_lon = float(matrix_df["GPS_0020_Lon"].max())

    lat_min = st.sidebar.number_input("Min Latitude",value=min_lat,format="%.6f")
    lat_max = st.sidebar.number_input("Max Latitude",value=max_lat,format="%.6f")

    lon_min = st.sidebar.number_input("Min Longitude",value=min_lon,format="%.6f")
    lon_max = st.sidebar.number_input("Max Longitude",value=max_lon,format="%.6f")

# -----------------------------
# FILTER DATA
# -----------------------------
    df = matrix_df.copy()

    df = df[df["GPS_0020_Lat"].between(lat_min,lat_max)]
    df = df[df["GPS_0020_Lon"].between(lon_min,lon_max)]

    plot_df = df.dropna(subset=["GPS_0020_Lat","GPS_0020_Lon",sel_sub])

# -----------------------------
# SESSION DATA
# -----------------------------
    if "active_df" not in st.session_state:
        st.session_state.active_df = plot_df.copy()

    if st.sidebar.button("Reset Dataset"):
        st.session_state.active_df = plot_df.copy()

    data = st.session_state.active_df.copy()

# -----------------------------
# AI CONTROLS
# -----------------------------
    st.sidebar.subheader("AI Analysis")

    run_cluster = st.sidebar.checkbox("Cluster Detection")
    run_anomaly = st.sidebar.checkbox("Anomaly Detection")

# -----------------------------
# CLUSTER
# -----------------------------
    if run_cluster and len(data) > 10:

        coords = data[["GPS_0020_Lat","GPS_0020_Lon"]]

        scaler = StandardScaler()

        X = scaler.fit_transform(coords)

        db = DBSCAN(eps=0.3,min_samples=5).fit(X)

        data["cluster"] = db.labels_

    else:
        data["cluster"] = -1

# -----------------------------
# ANOMALY
# -----------------------------
    if run_anomaly:

        z = np.abs((data[sel_sub]-data[sel_sub].mean())/data[sel_sub].std())

        data["anomaly"] = z > 2.5

    else:
        data["anomaly"] = False

# -----------------------------
# METRICS
# -----------------------------
    st.title(project_name)

    c1,c2,c3 = st.columns(3)

    c1.metric("Points",len(data))
    c2.metric("Average",round(data[sel_sub].mean(),3))
    c3.metric("Max",round(data[sel_sub].max(),3))

# -----------------------------
# TABS
# -----------------------------
    tab1,tab2,tab3,tab4,tab5 = st.tabs([
        "Satellite Map",
        "Polygon Tool",
        "Heatmap",
        "Contour",
        "3D"
    ])

# -----------------------------
# SATELLITE MAP
# -----------------------------
    with tab1:

        fig = px.scatter_map(
            data,
            lat="GPS_0020_Lat",
            lon="GPS_0020_Lon",
            color=sel_sub,
            zoom=19,
            height=750
        )

        fig.update_layout(
            map_style="white-bg",
            map_layers=[
                {
                    "below":"traces",
                    "sourcetype":"raster",
                    "source":[
                        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
                    ]
                }
            ]
        )

        st.plotly_chart(fig,use_container_width=True)

# -----------------------------
# POLYGON TOOL
# -----------------------------
    with tab2:

        center = [
            data["GPS_0020_Lat"].mean(),
            data["GPS_0020_Lon"].mean()
        ]

        m = folium.Map(location=center,zoom_start=19)

        Draw(export=True).add_to(m)

        st_folium(m,width=900,height=600)

# -----------------------------
# HEATMAP
# -----------------------------
    with tab3:

        heat = px.density_map(
            data,
            lat="GPS_0020_Lat",
            lon="GPS_0020_Lon",
            z=sel_sub,
            radius=15,
            zoom=18,
            height=750
        )

        st.plotly_chart(heat,use_container_width=True)

# -----------------------------
# CONTOUR
# -----------------------------
    with tab4:

        grid_x,grid_y = np.mgrid[
            data["GPS_0020_Lon"].min():data["GPS_0020_Lon"].max():200j,
            data["GPS_0020_Lat"].min():data["GPS_0020_Lat"].max():200j
        ]

        grid_z = griddata(
            (data["GPS_0020_Lon"],data["GPS_0020_Lat"]),
            data[sel_sub],
            (grid_x,grid_y),
            method="linear"
        )

        fig = px.imshow(grid_z,origin="lower")

        st.plotly_chart(fig,use_container_width=True)

# -----------------------------
# 3D
# -----------------------------
    with tab5:

        fig3d = px.scatter_3d(
            data,
            x="GPS_0020_Lon",
            y="GPS_0020_Lat",
            z=sel_sub,
            color=sel_sub
        )

        st.plotly_chart(fig3d,use_container_width=True)

# -----------------------------
# EXPORT
# -----------------------------
    csv = data.to_csv(index=False).encode()

    st.sidebar.download_button(
        "Download Cleaned CSV",
        csv,
        "filtered_data.csv",
        "text/csv"
    )

else:
    st.info("Upload a JSON file to start analysis.")
