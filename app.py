import streamlit as st
import joblib
import pickle
import pandas as pd
import matplotlib.pyplot as plt
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
# CONFIGURACIÓN
# ==========================================

st.set_page_config(
    page_title="Predicción de Residuos",
    page_icon="🗑️",
    layout="wide"
)

# ==========================================
# CONEXIÓN SUPABASE
# ==========================================

@st.cache_resource
def conectar_bd():

    connection = psycopg2.connect(
        user=USER,
        password=PASSWORD,
        host=HOST,
        port=PORT,
        dbname=DBNAME
    )

    return connection

# ==========================================
# CARGAR MODELO
# ==========================================

@st.cache_resource
def load_models():

    model = joblib.load("residuos_model.pkl")

    with open("residuos_model_info.pkl",
        "rb"
    ) as f:

        model_info = pickle.load(f)

    return model, model_info

# ==========================================
# LEER DATASET DESDE SUPABASE
# ==========================================

@st.cache_data
def cargar_datos():

    connection = conectar_bd()

    query = """
    SELECT *
    FROM pc_ml_residuos
    """

    df = pd.read_sql(query, connection)

    connection.close()

    return df

# ==========================================
# CONEXIÓN
# ==========================================

try:

    connection = conectar_bd()

    cursor = connection.cursor()

    cursor.execute("SELECT NOW();")

    result = cursor.fetchone()

    cursor.close()
    connection.close()

    st.sidebar.success(
        "Conectado a Supabase"
    )

except Exception as e:

    st.sidebar.error(
        f"Error conexión: {e}"
    )

# ==========================================
# CARGAR MODELO Y DATASET
# ==========================================

model, model_info = load_models()

df = cargar_datos()

# ==========================================
# TÍTULO
# ==========================================

st.title(
    "🗑️ Predicción de Residuos Municipales"
)

st.write(
    "Sistema Inteligente utilizando XGBoost y Supabase"
)

# ==========================================
# MENÚ
# ==========================================

menu = st.sidebar.selectbox(
    "Seleccionar Vista",
    [
        "Predicción",
        "Exploración Dataset",
        "Gráficos",
        "Top Departamentos",
        "Correlación"
    ]
)

# ==========================================
# PREDICCIÓN
# ==========================================

if menu == "Predicción":

    st.header(
        "Ingrese los datos"
    )

    inputs = {}

    for feature in model_info["columnas"]:

        valor = st.number_input(
            feature,
            value=0.0,
            step=1.0
        )

        inputs[feature] = valor

    if st.button("Predecir"):

        entrada = pd.DataFrame([inputs])

        prediction = model.predict(entrada)[0]

        st.success(
            f"Residuos Estimados: "
            f"{prediction:,.2f}"
        )

# ==========================================
# EXPLORACIÓN DATASET
# ==========================================

elif menu == "Exploración Dataset":

    st.header(
        "Exploración del Dataset"
    )

    st.subheader("Primeros registros")

    st.dataframe(df.head(20))

    st.subheader("Estadísticas")

    st.write(df.describe())

    st.subheader("Valores Nulos")

    st.write(df.isnull().sum())

    # --------------------------------------
    # FILTRO
    # --------------------------------------

    if "departamento" in df.columns:

        departamentos = sorted(
            df["departamento"]
            .dropna()
            .unique()
        )

        dep = st.selectbox(
            "Filtrar por departamento",
            departamentos
        )

        filtro = df[
            df["departamento"] == dep
        ]

        st.dataframe(filtro)

# ==========================================
# GRÁFICOS
# ==========================================

elif menu == "Gráficos":

    st.header(
        "Visualización de Datos"
    )

    # --------------------------------------
    # DISPERSIÓN
    # --------------------------------------

    if (
        "POB_TOTAL" in df.columns and
        "prediction" in df.columns
    ):

        st.subheader(
            "Población Total vs Predicción"
        )

        fig, ax = plt.subplots(figsize=(8,6))

        ax.scatter(
            df["POB_TOTAL"],
            df["prediction"]
        )

        ax.set_xlabel(
            "Población Total"
        )

        ax.set_ylabel(
            "Predicción"
        )

        st.pyplot(fig)

    # --------------------------------------
    # RESIDUOS POR PERIODO
    # --------------------------------------

    if (
        "PERIODO" in df.columns and
        "prediction" in df.columns
    ):

        st.subheader(
            "Predicciones por Año"
        )

        residuos_anio = (
            df.groupby("PERIODO")
            ["prediction"]
            .mean()
        )

        fig2, ax2 = plt.subplots(
            figsize=(8,6)
        )

        ax2.plot(
            residuos_anio.index,
            residuos_anio.values,
            marker='o'
        )

        ax2.set_xlabel("Año")
        ax2.set_ylabel("Predicción")

        st.pyplot(fig2)

# ==========================================
# TOP DEPARTAMENTOS
# ==========================================

elif menu == "Top Departamentos":

    st.header(
        "Top Departamentos"
    )

    if (
        "departamento" in df.columns and
        "prediction" in df.columns
    ):

        top_departamentos = (
            df.groupby("departamento")
            ["prediction"]
            .sum()
            .sort_values(
                ascending=False
            )
            .head(10)
        )

        st.bar_chart(
            top_departamentos
        )

        st.dataframe(
            top_departamentos
        )

# ==========================================
# CORRELACIÓN
# ==========================================

elif menu == "Correlación":

    st.header(
        "Correlación"
    )

    columnas = [
        'POB_TOTAL',
        'POB_URBANA',
        'POB_RURAL',
        'prediction'
    ]

    columnas_existentes = [
        c for c in columnas
        if c in df.columns
    ]

    correlacion = (
        df[columnas_existentes]
        .corr()
    )

    st.dataframe(
        correlacion
    )

    fig, ax = plt.subplots(
        figsize=(8,6)
    )

    cax = ax.matshow(
        correlacion
    )

    plt.xticks(
        range(len(correlacion.columns)),
        correlacion.columns,
        rotation=45
    )

    plt.yticks(
        range(len(correlacion.columns)),
        correlacion.columns
    )

    fig.colorbar(cax)

    st.pyplot(fig)
