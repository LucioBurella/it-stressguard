from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="IT-StressGuard",
    page_icon="🧠",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "modelo_it_stressguard.joblib"
DATA_PATH = BASE_DIR / "dataset_it_stressguard.csv"
METRICS_PATH = BASE_DIR / "modelo_it_stressguard.metricas.json"

RECOMENDACIONES = {
    "Bajo": "Mantener hábitos saludables, pausas regulares y organización de tareas.",
    "Medio": "Revisar prioridades, distribuir la carga y programar pausas durante la jornada.",
    "Alto": "Conversar con el líder sobre la carga de trabajo y evaluar una redistribución de tareas.",
    "Burnout Inminente": "Priorizar el descanso y solicitar acompañamiento del equipo responsable de bienestar.",
}


@st.cache_resource
def load_artifact(path: Path):
    if not path.exists():
        return None
    loaded = joblib.load(path)
    return loaded if isinstance(loaded, dict) else {"modelo": loaded}


@st.cache_data
def load_dataset(path: Path):
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data
def load_metrics(path: Path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def get_model(artifact: dict | None):
    if not artifact:
        return None
    return artifact.get("modelo")


def get_classes(model, artifact: dict | None) -> list[str]:
    if model is not None and hasattr(model, "classes_"):
        return [str(c) for c in model.classes_]
    if model is not None and hasattr(model, "named_steps") and "clasificador" in model.named_steps:
        clf = model.named_steps["clasificador"]
        if hasattr(clf, "classes_"):
            return [str(c) for c in clf.classes_]
    if artifact and isinstance(artifact.get("clases"), (list, tuple)):
        return [str(c) for c in artifact["clases"]]
    return ["Bajo", "Medio", "Alto", "Burnout Inminente"]


artifact = load_artifact(MODEL_PATH)
model = get_model(artifact)
df = load_dataset(DATA_PATH)
metrics_file = load_metrics(METRICS_PATH)

# --- Barra lateral de navegación ---
st.sidebar.title("IT-StressGuard")
st.sidebar.caption("Prototipo académico de análisis y predicción de estrés laboral")

menu = st.sidebar.radio(
    "Menú principal",
    [
        "Predicción",
        "Exploración del Dataset",
        "Modelo y Métricas",
    ],
)

st.sidebar.divider()
st.sidebar.info(
    "Nota: Prototipo académico con datos sintéticos; no reemplaza una evaluación profesional."
)

# ==============================================================================
# MENÚ 1: PREDICCIÓN
# ==============================================================================
if menu == "Predicción":
    st.title("Predicción de Nivel de Estrés Laboral")
    st.write("Completá los indicadores de la jornada para estimar el nivel de estrés y ver su distribución probabilística.")

    if model is None:
        st.error("No se encontró `modelo_it_stressguard.joblib` o no contiene la clave `'modelo'`.")
        st.stop()

    with st.form("form_prediccion"):
        col1, col2, col3 = st.columns(3)
        with col1:
            horas = st.slider("Horas trabajadas", min_value=0.0, max_value=24.0, value=8.0, step=0.5)
        with col2:
            tickets = st.number_input("Tickets resueltos", min_value=0, max_value=500, value=6, step=1)
        with col3:
            fatiga = st.slider("Fatiga percibida (1 a 10)", min_value=1, max_value=10, value=5, step=1)

        col4, col5 = st.columns([1, 2])
        with col4:
            rol = st.selectbox("Rol", options=["Dev", "DevOps", "QA", "Team Lead", "HR"], index=0)
        with col5:
            comentario = st.text_input(
                "Comentario del día (opcional)",
                placeholder="Ej: Día con varios despliegues y revisiones de código...",
            )

        submitted = st.form_submit_button("Analizar y Predecir", type="primary", use_container_width=True)

    if submitted:
        entrada = pd.DataFrame(
            [
                {
                    "horasTrabajadas": horas,
                    "ticketsResueltos": tickets,
                    "nivelFatiga": fatiga,
                    "comentarioDia": comentario,
                    "rol": rol,
                }
            ]
        )

        try:
            nivel = str(model.predict(entrada)[0])
            probabilidades = model.predict_proba(entrada)[0] if hasattr(model, "predict_proba") else None
        except Exception as exc:
            st.error(f"Error al ejecutar la predicción: {exc}")
            st.stop()

        st.divider()
        st.subheader("Resultado del análisis")

        m1, m2, m3 = st.columns(3)
        m1.metric("Nivel de estrés estimado", nivel)
        if probabilidades is not None:
            prob_max = float(max(probabilidades))
            m2.metric("Confianza máxima", f"{prob_max:.1%}")
        m3.metric("Carga declarada", f"{horas:.1f}h / {tickets} tickets")

        st.success(f"**Recomendación:** {RECOMENDACIONES.get(nivel, 'Revisar el resultado con un profesional.')}")

        # Resumen de factores clave de la entrada
        with st.expander("Detalle de variables enviadas", expanded=False):
            st.dataframe(entrada, use_container_width=True, hide_index=True)

        if probabilidades is not None:
            clases = get_classes(model, artifact)
            if len(clases) != len(probabilidades):
                clases = [f"Clase {i}" for i in range(len(probabilidades))]

            prob_df = (
                pd.DataFrame({"Clase": clases, "Probabilidad": [float(p) for p in probabilidades]})
                .sort_values("Probabilidad", ascending=False)
                .reset_index(drop=True)
            )

            g1, g2 = st.columns([3, 2])
            with g1:
                st.markdown("**Probabilidad por nivel de estrés**")
                st.bar_chart(prob_df.set_index("Clase")["Probabilidad"])
            with g2:
                st.markdown("**Tabla de probabilidades**")
                st.dataframe(
                    prob_df.style.format({"Probabilidad": "{:.2%}"}),
                    use_container_width=True,
                    hide_index=True,
                )

# ==============================================================================
# MENÚ 2: EXPLORACIÓN DEL DATASET
# ==============================================================================
elif menu == "Exploración del Dataset":
    st.title("Exploración del Dataset")

    if df is None or df.empty:
        st.warning("No se encontró `dataset_it_stressguard.csv` o está vacío.")
        st.stop()

    k1, k2, k3 = st.columns(3)
    k1.metric("Total de registros", len(df))
    k2.metric("Columnas", len(df.columns))
    k3.metric("Valores nulos", int(df.isna().sum().sum()))

    st.divider()

    # Detectar columna objetivo si existe
    target_candidates = [c for c in df.columns if c.lower() in {"nivelestres", "nivel_estres", "estres", "stress", "target"}]
    target_col = target_candidates[0] if target_candidates else None

    c1, c2 = st.columns(2)

    with c1:
        if target_col:
            st.subheader(f"Distribución de `{target_col}`")
            counts = df[target_col].astype(str).value_counts()
            st.bar_chart(counts)
        else:
            st.subheader("Columnas disponibles")
            st.write(list(df.columns))

    with c2:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if numeric_cols:
            selected_num = st.selectbox("Variable numérica para analizar", options=numeric_cols)
            st.subheader(f"Distribución de `{selected_num}`")
            st.line_chart(df[selected_num].dropna().reset_index(drop=True))

    st.divider()
    st.subheader("Vista previa de datos")
    st.dataframe(df.head(50), use_container_width=True)

# ==============================================================================
# MENÚ 3: MODELO Y MÉTRICAS
# ==============================================================================
elif menu == "Modelo y Métricas":
    st.title("Modelo y Métricas")

    version = artifact.get("version", "desconocida") if artifact else "no disponible"
    algoritmo = artifact.get("algoritmo", "RandomForestClassifier + TF-IDF") if artifact else "no disponible"
    clases = get_classes(model, artifact)

    i1, i2, i3 = st.columns(3)
    i1.metric("Versión del artefacto", str(version))
    i2.metric("Algoritmo", str(algoritmo))
    i3.metric("Cantidad de clases", len(clases))

    st.markdown("**Clases detectadas:** " + ", ".join(f"`{c}`" for c in clases))

    st.divider()

    # Unificar métricas desde archivo .metricas.json o desde el diccionario del modelo
    metricas_dict = {}
    if isinstance(metrics_file, dict):
        metricas_dict.update(metrics_file)
    if artifact and isinstance(artifact.get("metricas_prueba"), dict):
        for k, v in artifact["metricas_prueba"].items():
            metricas_dict.setdefault(k, v)

    if metricas_dict:
        st.subheader("Indicadores de rendimiento")
        cols = st.columns(min(len(metricas_dict), 4))
        for idx, (k, v) in enumerate(metricas_dict.items()):
            label = k.replace("_", " ").capitalize()
            val_fmt = f"{v:.4f}" if isinstance(v, float) else str(v)
            cols[idx % len(cols)].metric(label=label, value=val_fmt)

        # Gráfico con métricas numéricas (excluyendo conteos grandes si distorsionan la escala)
        rate_metrics = {
            k: float(v)
            for k, v in metricas_dict.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool) and 0.0 <= float(v) <= 1.0
        }
        if rate_metrics:
            st.markdown("**Métricas en escala 0–1**")
            st.bar_chart(pd.Series(rate_metrics, name="Valor"))

        with st.expander("Ver JSON completo de métricas"):
            st.json(metricas_dict)
    else:
        st.info("No se encontraron métricas guardadas en `modelo_it_stressguard.metricas.json` ni en el `.joblib`.")