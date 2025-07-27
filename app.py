import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from shapely.geometry import Point
from scipy.spatial import cKDTree
from sklearn.metrics import precision_score, recall_score, f1_score
import numpyro.distributions as dist
from bstpp.main import Hawkes_Model

# Configuración general
st.set_page_config(page_title="Predicción de Hurtos con Hawkes", layout="wide")

st.title("🔍 Predicción de Hurtos con Modelo Hawkes")
st.markdown("Esta app entrena un modelo Hawkes sobre datos de hurtos y simula eventos futuros.")

# Cargar archivos
st.sidebar.header("1️⃣ Cargar datos")
events_file = st.sidebar.file_uploader("Archivo de hurtos (.shp + otros)", type=["shp"])
boundaries_file = st.sidebar.file_uploader("Archivo de red vial (.shp + otros)", type=["shp"])

if events_file and boundaries_file:
    # Cargar datos (requiere subir .shp + .shx + .dbf + .prj)
    st.success("✅ Archivos cargados correctamente. Procesando...")

    gdf_events = gpd.read_file(events_file).to_crs("EPSG:4326")
    gdf_boundaries = gpd.read_file(boundaries_file).to_crs("EPSG:4326")
    
    gdf_events["Fecha"] = pd.to_datetime(gdf_events["Fecha"], format="%d/%m/%Y", errors="coerce")
    gdf_events["t"] = (gdf_events["Fecha"] - gdf_events["Fecha"].min()).dt.total_seconds() / 86400
    gdf_events = gdf_events.sort_values(by="t").reset_index(drop=True)

    fecha_split = pd.to_datetime("2018-12-31")
    gdf_train = gdf_events[gdf_events["Fecha"] < fecha_split].copy()
    gdf_test = gdf_events[gdf_events["Fecha"] >= fecha_split].copy()

    # Crear buffer
    gdf_buffered = gdf_boundaries.copy()
    gdf_buffered["geometry"] = gdf_buffered.buffer(0.0002)

    # Reescalar tiempo
    T_real = gdf_train["t"].max()
    t_extra = (gdf_test["t"].max() - T_real)
    T_model = T_real + t_extra
    gdf_train["t_scaled"] = gdf_train["t"] * (50.0 / T_model)
    data_model = gdf_train[["t_scaled", "Long", "Lat"]].rename(columns={"t_scaled": "T", "Long": "X", "Lat": "Y"})

    # Entrenar modelo
    if st.sidebar.button("🚀 Entrenar modelo Hawkes"):
        with st.spinner("Entrenando modelo..."):
            model = Hawkes_Model(
                data=data_model,
                A=gdf_buffered,
                T=50.0,
                cox_background=True,
                a_0=dist.Normal(1, 10),
                alpha=dist.Beta(20, 60),
                beta=dist.HalfNormal(2.0),
                sigmax_2=dist.HalfNormal(0.25)
            )
            model.run_svi(lr=0.02, num_steps=2000)
        st.success("✅ Modelo entrenado.")

        # Visualizaciones
        st.header("📊 Visualización del modelo")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Intensidad espacial")
            fig1 = model.plot_spatial(include_cov=False)
            st.pyplot(fig1)

        with col2:
            st.subheader("Proporción de autoexcitación")
            fig2 = model.plot_prop_excitation()
            st.pyplot(fig2)

        # Simular eventos
        st.subheader("🔮 Simulación de eventos futuros")
        sim_events = model.simulate()
        sim_events["T_real"] = sim_events["T"] * (T_model / 50.0)
        t0 = gdf_events["Fecha"].min()
        sim_events["Fecha_simulada"] = t0 + pd.to_timedelta(sim_events["T_real"], unit="D")

        # Filtrar
        sim_futuro = sim_events[sim_events["T_real"] > T_real]
        gdf_sim = gpd.GeoDataFrame(sim_futuro, geometry=gpd.points_from_xy(sim_futuro["X"], sim_futuro["Y"]), crs="EPSG:4326")

        st.map(gdf_sim)

        st.info(f"🔮 Se simularon {len(gdf_sim)} eventos futuros.")

        # Métrica de comparación
        st.subheader("📏 Comparación con eventos reales")
        sim_coords = np.array([[p.x, p.y] for p in gdf_sim.geometry])
        real_coords = np.array([[p.x, p.y] for p in gdf_test.geometry])
        tree = cKDTree(sim_coords)
        matches = tree.query_ball_point(real_coords, r=0.002)
        y_pred = np.array([1 if len(m) > 0 else 0 for m in matches])
        y_true = np.ones(len(real_coords))

        recall = recall_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        st.metric("📈 Recall (cobertura)", f"{recall:.2f}")
        st.metric("📍 Precisión espacial", f"{precision:.2f}")
        st.metric("📊 F1-score", f"{f1:.2f}")

else:
    st.warning("📁 Por favor, sube ambos archivos de entrada (.shp). Asegúrate de incluir todos los archivos relacionados (.shx, .dbf, .prj).")

