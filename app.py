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
    "Sistema Inteligente utilizando XGBoost y Datos Abiertos del Perú"
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
# CARGAR MODELO
# ==========================================

model, model_info = load_models()

# ==========================================
# VISTA PREDICCIÓN
# ==========================================

if menu == "Predicción":

    if model is not None:

        st.header(
            "Ingrese los datos del distrito"
        )

        inputs = {}

        for feature in model_info["columnas"]:

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

# ==========================================
# EXPLORACIÓN DATASET
# ==========================================

elif menu == "Exploración Dataset":

    st.header("Exploración del Dataset")

    st.subheader("Primeras filas")

    st.dataframe(df.head(20))

    st.subheader("Información General")

    st.write(df.describe())

    st.subheader("Valores Nulos")

    st.write(df.isnull().sum())

    st.subheader("Filtrar por Departamento")

    departamentos = sorted(
        df['DEPARTAMENTO'].unique()
    )

    dep = st.selectbox(
        "Seleccione departamento",
        departamentos
    )

    filtro = df[
        df['DEPARTAMENTO'] == dep
    ]

    st.dataframe(filtro.head(50))

# ==========================================
# GRÁFICOS
# ==========================================

elif menu == "Gráficos":

    st.header("Visualización de Datos")

    # --------------------------------------
    # Población vs Residuos
    # --------------------------------------

    st.subheader(
        "Población Total vs Residuos"
    )

    fig, ax = plt.subplots(figsize=(8,6))

    ax.scatter(
        df['POB_TOTAL'],
        df['QRESIDUOS_MUN']
    )

    ax.set_xlabel("Población Total")
    ax.set_ylabel("Residuos")

    st.pyplot(fig)

    # --------------------------------------
    # Residuos por Año
    # --------------------------------------

    st.subheader(
        "Residuos por Año"
    )

    residuos_anio = (
        df.groupby('PERIODO')
        ['QRESIDUOS_MUN']
        .sum()
    )

    fig2, ax2 = plt.subplots(figsize=(8,6))

    ax2.plot(
        residuos_anio.index,
        residuos_anio.values,
        marker='o'
    )

    ax2.set_xlabel("Año")
    ax2.set_ylabel("Residuos")

    st.pyplot(fig2)

# ==========================================
# TOP DEPARTAMENTOS
# ==========================================

elif menu == "Top Departamentos":

    st.header(
        "Departamentos con Más Residuos"
    )

    top_departamentos = (
        df.groupby('DEPARTAMENTO')
        ['QRESIDUOS_MUN']
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    st.bar_chart(top_departamentos)

    st.dataframe(top_departamentos)

# ==========================================
# CORRELACIÓN
# ==========================================

elif menu == "Correlación":

    st.header("Correlación entre Variables")

    correlacion = df[[
        'POB_TOTAL',
        'POB_URBANA',
        'POB_RURAL',
        'QRESIDUOS_MUN'
    ]].corr()

    st.dataframe(correlacion)

    fig, ax = plt.subplots(figsize=(8,6))

    cax = ax.matshow(correlacion)

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
