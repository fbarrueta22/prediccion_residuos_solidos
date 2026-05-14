import streamlit as st
import joblib
import pickle
import pandas as pd
import psycopg2

# ==========================================
# SECRETS
# ==========================================

USER = st.secrets["postgres"]["USER"]
PASSWORD = st.secrets["postgres"]["PASSWORD"]
HOST = st.secrets["postgres"]["HOST"]
PORT = st.secrets["postgres"]["PORT"]
DBNAME = st.secrets["postgres"]["DBNAME"]

# ==========================================
# CONFIG
# ==========================================

st.set_page_config(
    page_title="Predicción de Residuos",
    page_icon="🗑️"
)

# ==========================================
# CONEXIÓN BD
# ==========================================

try:

    connection = psycopg2.connect(
        user=USER,
        password=PASSWORD,
        host=HOST,
        port=PORT,
        dbname=DBNAME
    )

    cursor = connection.cursor()

    cursor.execute("SELECT NOW();")

    result = cursor.fetchone()

    cursor.close()
    connection.close()

    st.sidebar.success(
        f"Conectado a Supabase: {result}"
    )

except Exception as e:

    st.sidebar.error(
        f"Error de conexión: {e}"
    )

# ==========================================
# CARGAR MODELO
# ==========================================

@st.cache_resource
def load_models():

    try:

        model = joblib.load(
            "residuos_model.pkl"
        )

        with open(
            "residuos_model_info.pkl",
            "rb"
        ) as f:

            model_info = pickle.load(f)

        return model, model_info

    except FileNotFoundError:

        st.error(
            "No se encontraron los archivos del modelo"
        )

        return None, None

# ==========================================
# TÍTULO
# ==========================================

st.title(
    "🗑️ Predicción de Residuos Municipales"
)

st.write(
    "Sistema Inteligente con XGBoost"
)

# ==========================================
# CARGAR MODELO
# ==========================================

model, model_info = load_models()

# ==========================================
# INPUTS
# ==========================================

if model is not None:

    st.header(
        "Ingrese los datos del distrito"
    )

    inputs = {}

    for feature in model_info["feature_names"]:

        valor = st.number_input(
            feature,
            value=0.0,
            step=1.0
        )

        inputs[feature] = valor

    # ==========================================
    # PREDICCIÓN
    # ==========================================

    if st.button("Predecir"):

        entrada = pd.DataFrame([inputs])

        prediction = model.predict(entrada)[0]

        st.success(
            f"Residuos Municipales Estimados: "
            f"{prediction:,.2f}"
        )