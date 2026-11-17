import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Riesgo de Crédito")

carpeta = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def cargar_modelos():
    escalador = joblib.load(os.path.join(carpeta, "modelos", "escalador.pkl"))
    columnas = joblib.load(os.path.join(carpeta, "modelos", "columnas.pkl"))
    orden_grado = joblib.load(os.path.join(carpeta, "modelos", "orden_grado.pkl"))
    tasa_real = joblib.load(os.path.join(carpeta, "modelos", "tasa_real.pkl"))
    umbral = joblib.load(os.path.join(carpeta, "modelos", "umbral.pkl"))
    archivos = joblib.load(os.path.join(carpeta, "modelos", "archivos_modelos.pkl"))
    modelos = {}
    for nombre in archivos:
        modelos[nombre] = joblib.load(os.path.join(carpeta, "modelos", archivos[nombre]))
    return escalador, columnas, orden_grado, tasa_real, umbral, modelos


escalador, columnas, orden_grado, tasa_real, umbral, modelos = cargar_modelos()

# ---------------- Barra lateral ----------------
st.sidebar.header("Datos del solicitante")
edad = st.sidebar.slider("Edad", 18, 100, 30)
ingreso = st.sidebar.slider("Ingreso anual en dólares", 4000, 500000, 50000, step=1000)
empleo = st.sidebar.slider("Años trabajando", 0, 60, 3)
vivienda = st.sidebar.selectbox("Situación de vivienda", ["RENT", "OWN", "MORTGAGE", "OTHER"])
incumplio_antes = st.sidebar.selectbox("¿Tiene incumplimientos anteriores?", ["N", "Y"])

st.sidebar.header("Datos del préstamo")
proposito = st.sidebar.selectbox("Propósito del préstamo", ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL",
                                                            "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"])
grado = st.sidebar.selectbox("Calificación del préstamo", ["A", "B", "C", "D", "E", "F", "G"])
monto = st.sidebar.slider("Monto solicitado en dólares", 500, 35000, 10000, step=500)
tasa = st.sidebar.slider("Tasa de interés anual en %", 5.0, 30.0, 11.0, step=0.1)

st.sidebar.header("Modelo")
opciones = ["Boosting", "Bagging", "Árbol de Decisión", "Votación", "SVM", "KNN", "Regresión Logística", "Naive Bayes"]
nombre_modelo = st.sidebar.selectbox("Seleccionar modelo", opciones)
st.sidebar.caption("Boosting es el modelo final del notebook. Usa un umbral de " + str(umbral) + " para detectar al menos 8 de cada 10 clientes que incumplen.")

predecir = st.sidebar.button("Predecir")

# ---------------- Página principal ----------------
st.title("Predicción del Riesgo de Crédito")
st.write("Esta aplicación predice si un solicitante va a pagar o a incumplir su crédito, según sus datos "
         "personales, los datos del préstamo y el modelo seleccionado. Los datos se llenan en la barra de la izquierda.")

porcentaje = round(monto / ingreso, 2)

datos = pd.DataFrame({
    "person_age": [edad],
    "person_income": [ingreso],
    "person_home_ownership": [vivienda],
    "person_emp_length": [empleo],
    "loan_intent": [proposito],
    "loan_grade": [grado],
    "loan_amnt": [monto],
    "loan_int_rate": [tasa],
    "loan_percent_income": [porcentaje],
    "cb_person_default_on_file": [incumplio_antes]
})

st.subheader("Datos ingresados")
st.dataframe(datos, hide_index=True)

if empleo > edad - 14:
    st.warning("Los años trabajando no pueden ser mayores a la edad menos 14.")

if predecir:
    entrada = datos.copy()
    entrada["loan_grade"] = entrada["loan_grade"].map(orden_grado)
    entrada["cb_person_default_on_file"] = entrada["cb_person_default_on_file"].map({"Y": 1, "N": 0})
    entrada = pd.get_dummies(entrada, columns=["person_home_ownership", "loan_intent"], dtype=int)
    entrada = entrada.reindex(columns=columnas, fill_value=0)
    entrada = escalador.transform(entrada)

    modelo = modelos[nombre_modelo]
    if nombre_modelo == "Boosting":
        limite = umbral
        prediccion = int(modelo.predict_proba(entrada)[0][1] >= umbral)
    else:
        limite = 0.5
        prediccion = modelo.predict(entrada)[0]

    st.subheader("Resultado")
    st.write("Modelo usado:", nombre_modelo)

    if prediccion == 1:
        st.error("La predicción es: el cliente va a INCUMPLIR")
    else:
        st.success("La predicción es: el cliente va a PAGAR")

    if hasattr(modelo, "predict_proba"):
        probabilidad = modelo.predict_proba(entrada)[0][1]
        if nombre_modelo == "KNN":
            st.metric("Probabilidad de incumplir", str(round(probabilidad * 100, 1)) + " %")
            st.progress(float(probabilidad))
        else:
            ajustada = (probabilidad * tasa_real) / (probabilidad * tasa_real + (1 - probabilidad) * (1 - tasa_real))
            limite_ajustado = (limite * tasa_real) / (limite * tasa_real + (1 - limite) * (1 - tasa_real))
            st.metric("Probabilidad real de incumplir", str(round(ajustada * 100, 1)) + " %")
            st.progress(float(ajustada))
            st.caption("El modelo se entrenó dando el mismo peso a los clientes que pagan y a los que incumplen, "
                       "por lo que su probabilidad directa fue de " + str(round(probabilidad * 100, 1)) + " %. "
                       "La probabilidad real se ajusta con la tasa de incumplimiento de los datos, que es de "
                       + str(round(tasa_real * 100, 1)) + " %. Con este modelo se marca como riesgoso a quien pase de "
                       + str(round(limite_ajustado * 100, 1)) + " % de probabilidad real.")
    else:
        st.info("Este modelo no calcula probabilidades, solo la clase.")
else:
    st.info("Llene los datos en la barra lateral y oprima Predecir.")
