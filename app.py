import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import json

import numpyro.distributions as dist
from bstpp.main import Hawkes_Model

st.title("Simulación de riesgo espacial de eventos delictivos con Hawkes_Model")

# 1. Subida de datos (GeoJSON preferido para simplicidad)
uploaded_events = st.file_uploader("Sube archivo GeoJSON de eventos (con campo 'Fecha', 'Long', 'Lat')", type=["geojson", "json"])
uploaded_boundaries = st.file_uploader("Sube GeoJSON con red vial (líneas) para buffer", type=["geojson", "json"])

if uploaded_events is not None and uploaded_boundaries is not None:
    # Leer eventos
    events_json = json.load(uploaded_events)
    # Extraer geometría y propiedades
    features = events_json["features"]
    
    rows = []
    for f in features:
        props = f["properties"]
        geom = f["geometry"]
        # Asumo geom tipo Point con coords [long, lat]
        rows.append({
            "Long": geom["coordinates"][0],
            "Lat": geom["coordinates"][1],
            "Fecha": props.get("Fecha")
        })
    df_events = pd.DataFrame(rows)
    
    # Procesar fechas
    df_events["Fecha"] = pd.to_datetime(df_events["Fecha"], format="%d/%m/%Y", errors="coerce")
    df_events = df_events.dropna(subset=["Fecha"])
    
    # Temporal en días desde t0
    t0 = df_events["Fecha"].min()
    df_events["t"] = (df_events["Fecha"] - t0).dt.total_seconds() / 86400
    df_events = df_events.sort_values(by="t").reset_index(drop=True)
    
    # Leer boundaries (líneas)
    boundaries_json = json.load(uploaded_boundaries)
    # Aquí deberías hacer un buffer sobre líneas (si tienes shapely, usa shapely.geometry.LineString.buffer)
    # Pero shapely puede ser pesado, así que si no quieres dependencias, omite el buffer y usa raw lines
    
    # Mostrar opciones y parámetros
    fecha_split = st.date_input("Fecha de corte train/test", value=pd.to_datetime("2018-12-31"))
    
    # Split
    df_train = df_events[df_events["Fecha"] < pd.to_datetime(fecha_split)]
    df_test = df_events[df_events["Fecha"] >= pd.to_datetime(fecha_split)]
    
    # Preparar datos para el modelo
    data_model = df_train[["t", "Long", "Lat"]].rename(columns={"t": "T", "Long": "X", "Lat": "Y"})
    
    st.write("Entrenando modelo...")
    
    # Aquí deberías crear el buffer si usas shapely (opcional)
    # Por simplicidad, sin buffer, pasamos None o raw boundaries
    model = Hawkes_Model(
        data=data_model,
        A=None,  # Poner boundaries procesados si quieres
        T=df_train["t"].max(),
        cox_background=True,
        a_0=dist.Normal(1, 10),
        alpha=dist.Beta(20, 60),
        beta=dist.HalfNormal(2.0),
        sigmax_2=dist.HalfNormal(0.25)
    )
    
    model.run_svi(lr=0.02, num_steps=500)  # Menos pasos para demo
    
    st.success("Modelo entrenado.")
    
    # Evaluación
    data_test = df_test[["t", "Long", "Lat"]].rename(columns={"t": "T", "Long": "X", "Lat": "Y"})
    
    st.write("Evaluación:")
    st.write(f"Log Expected Likelihood: {model.log_expected_likelihood(data_test)}")
    st.write(f"Expected AIC: {model.expected_AIC()}")
    
    # Visualización simple con matplotlib
    fig, ax = plt.subplots(figsize=(10,6))
    model.plot_spatial(include_cov=False)
    plt.title("Intensidad espacial sobre la red vial (background Cox)")
    st.pyplot(fig)
    
    # Otros gráficos y explicaciones (puedes añadirlos aquí)
else:
    st.info("Por favor sube ambos archivos GeoJSON para continuar.")
