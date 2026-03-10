import streamlit as st
import json
import pandas as pd
import plotly.express as px
import numpy as np
from streamlit_plotly_events import plotly_events

st.set_page_config(page_title="Grave Analysis Dashboard", layout="wide")

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

    df['Time_Sec'] = (df['vT'] / 1000).round().astype(int)

    df = df.groupby('Time_Sec').first().reset_index()

    return df


# --- SIDEBAR ---
st.sidebar.title("🛠️ Precise Control Panel")

project_name = st.sidebar.text_input("Project Name", value="New Survey Site")

uploaded_file = st.sidebar.file_uploader("Upload .json data file", type=["json"])

if uploaded_file is not None:

    matrix_df = load_and_process_data(uploaded_file)

    if matrix_df is not None:

        st.sidebar.success("Data Loaded Successfully!")

        cols = [c for c in matrix_df.columns if 'STABSPECTRO' in c and not c.startswith('s')]

        sel_sub = st.sidebar.selectbox("Analyze Substance", cols)

        st.sidebar.divider()

        st.sidebar.subheader("📍 Area of Interest")

        min_lat_val = float(matrix_df['GPS_0020_Lat'].min())
        max_lat_val = float(matrix_df['GPS_0020_Lat'].max())

        in_lat_min = st.sidebar.number_input("Min Latitude", value=min_lat_val, format="%.6f")
        in_lat_max = st.sidebar.number_input("Max Latitude", value=max_lat_val, format="%.6f")

        min_lon_val = float(matrix_df['GPS_0020_Lon'].min())
        max_lon_val = float(matrix_df['GPS_0020_Lon'].max())

        in_lon_min = st.sidebar.number_input("Min Longitude", value=min_lon_val, format="%.6f")
        in_lon_max = st.sidebar.number_input("Max Longitude", value=max_lon_val, format="%.6f")

        st.sidebar.divider()

        st.sidebar.subheader("📏 Altitude & Speed")

        h_min = st.sidebar.number_input("Min Height (m)", value=0.0)
        h_max = st.sidebar.number_input("Max Height (m)", value=60.0)

        speed_col = 'GPS_0020_gSpeed'

        s_min = 0.0
        s_max = 10.0

        if speed_col in matrix_df.columns:
            s_max = float(matrix_df[speed_col].max())

            s_min = st.sidebar.number_input("Min Walking Speed", value=0.0)
            s_max = st.sidebar.number_input("Max Walking Speed", value=s_max)

        st.sidebar.subheader("✨ Noise Reduction")

        p_range = st.sidebar.slider("Concentration Percentile", 0, 100, (0, 100))

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

        if "active_df" not in st.session_state:
            st.session_state.active_df = plot_df.copy()

        st.title(f"⚰️ {project_name}")

        m1, m2, m3 = st.columns(3)

        m1.metric("Points Found", len(st.session_state.active_df))

        if not st.session_state.active_df.empty:

            m2.metric("Avg Concentration", f"{st.session_state.active_df[sel_sub].mean():.3f}")
            m3.metric("Max Concentration", f"{st.session_state.active_df[sel_sub].max():.3f}")

        else:
            m2.metric("Avg Concentration", "0")
            m3.metric("Max Concentration", "0")

        t1, t2, t3 = st.tabs(["🗺️ Satellite Map", "📈 3D View", "📋 Raw Data"])

        # --- SATELLITE MAP ---
        with t1:

            if not st.session_state.active_df.empty:

                fig_map = px.scatter_map(
                    st.session_state.active_df,
                    lat='GPS_0020_Lat',
                    lon='GPS_0020_Lon',
                    color=sel_sub,
                    zoom=19,
                    height=750,
                    color_continuous_scale="Viridis"
                )

                fig_map.update_layout(
                    map_style="white-bg",
                    dragmode="lasso",
                    map_layers=[{
                        "below": 'traces',
                        "sourcetype": "raster",
                        "source": ["https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"]
                    }]
                )

                selected_points = plotly_events(
                    fig_map,
                    select_event=True,
                    override_height=750
                )

                st.plotly_chart(fig_map, use_container_width=True)

                if selected_points:

                    st.warning(f"{len(selected_points)} points selected")

                    if st.button("❌ Remove Selected Points"):

                        idx = [p["pointIndex"] for p in selected_points]

                        st.session_state.active_df = st.session_state.active_df.drop(
                            st.session_state.active_df.index[idx]
                        ).reset_index(drop=True)

                        st.success("Points removed")

                        st.rerun()

        # --- 3D VIEW ---
        with t2:

            if not st.session_state.active_df.empty:

                fig_3d = px.scatter_3d(
                    st.session_state.active_df,
                    x='GPS_0020_Lon',
                    y='GPS_0020_Lat',
                    z=sel_sub,
                    color=sel_sub,
                    height=750,
                    color_continuous_scale="Viridis"
                )

                st.plotly_chart(fig_3d, use_container_width=True)

        # --- RAW DATA ---
        with t3:

            st.dataframe(st.session_state.active_df, use_container_width=True)

        # --- EXPORT ---
        st.sidebar.divider()

        csv_data = st.session_state.active_df.to_csv(index=False).encode('utf-8')

        st.sidebar.download_button(
            "Download Filtered CSV",
            csv_data,
            file_name="filtered_data.csv",
            mime="text/csv"
        )

else:

    st.info("👋 Upload your .json survey file to begin analysis.")
