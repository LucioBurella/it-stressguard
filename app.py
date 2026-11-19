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


@st.cache_resource
def load_artifact(path: Path):
    return joblib.load(path)


def extract_estimator(artifact):
    if hasattr(artifact, "predict"):
        return artifact
    if isinstance(artifact, dict):
        common_keys = ("model", "modelo", "pipeline", "estimator", "clf", "classifier")
        for key in common_keys:
            val = artifact.get(key)
            if val is not None and hasattr(val, "predict"):
                return val
        for val in artifact.values():
            if hasattr(val, "predict"):
                return val
    return None


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


def get_feature_names(estimator, artifact, df):
    for obj in (estimator, artifact):
        if hasattr(obj, "feature_names_in_"):
            return list(obj.feature_names_in_)
        if hasattr(obj, "named_steps"):
            for step in obj.named_steps.values():
                if hasattr(step, "feature_names_in_"):
                    return list(step.feature_names_in_)
        if isinstance(obj, dict):
            for k in ("feature_names", "features", "columns", "input_cols"):
                if isinstance(obj.get(k), (list, tuple)) and obj[k]:
                    return list(obj[k])

    if df is not None:
        excluded = {"target", "label", "stress", "estres", "nivel_estres", "nivelestres"}
        return [c for c in df.columns if c.lower() not in excluded]
    return []


st.title("IT-StressGuard")
st.caption("Predicción de nivel de estrés laboral — Prototipo interactivo con Streamlit")

if not MODEL_PATH.exists():
    st.error(f"No se encontró el modelo en `{MODEL_PATH}`.")
    st.stop()

artifact = load_artifact(MODEL_PATH)
model = extract_estimator(artifact)
df = load_dataset(DATA_PATH)
metrics = load_metrics(METRICS_PATH)

if model is None:
    st.error("El archivo `.joblib` es un diccionario pero no contiene un objeto con `.predict`.")
    if isinstance(artifact, dict):
        st.write("Claves encontradas en el `.joblib`:", list(artifact.keys()))
    st.stop()

if metrics:
    with st.expander("Métricas del modelo", expanded=True):
        cols = st.columns(len(metrics))
        for idx, (k, v) in enumerate(metrics.items()):
            label = k.replace("_", " ").capitalize()
            val_fmt = f"{v:.4f}" if isinstance(v, float) else str(v)
            cols[idx].metric(label=label, value=val_fmt)

st.divider()

feature_names = get_feature_names(model, artifact, df)
if not feature_names:
    st.warning("No se detectaron columnas de entrada.")
    st.stop()

if "sample_row" not in st.session_state:
    st.session_state.sample_row = None

top_bar_left, top_bar_right = st.columns([3, 1])
with top_bar_left:
    st.subheader("Datos de entrada")
with top_bar_right:
    if df is not None and not df.empty:
        if st.button("Cargar ejemplo del CSV", use_container_width=True):
            st.session_state.sample_row = df.sample(1).iloc[0].to_dict()
            st.rerun()

inputs = {}
with st.form("prediction_form"):
    grid_cols = st.columns(2)
    sample = st.session_state.sample_row or {}

    for i, col in enumerate(feature_names):
        target_col = grid_cols[i % 2]
        with target_col:
            if df is not None and col in df.columns:
                serie = df[col].dropna()
                sample_val = sample.get(col)

                if pd.api.types.is_numeric_dtype(serie):
                    min_val = float(serie.min()) if not serie.empty else 0.0
                    max_val = float(serie.max()) if not serie.empty else 100.0
                    default_num = (
                        float(sample_val)
                        if sample_val is not None and pd.notna(sample_val)
                        else (float(serie.median()) if not serie.empty else 0.0)
                    )
                    if pd.api.types.is_integer_dtype(serie):
                        inputs[col] = st.number_input(
                            col,
                            min_value=int(min_val),
                            max_value=int(max_val),
                            value=int(round(default_num)),
                            step=1,
                        )
                    else:
                        inputs[col] = st.number_input(
                            col,
                            min_value=min_val,
                            max_value=max_val,
                            value=default_num,
                        )
                else:
                    opciones = sorted(serie.astype(str).unique().tolist())
                    if opciones:
                        default_idx = 0
                        if sample_val is not None and str(sample_val) in opciones:
                            default_idx = opciones.index(str(sample_val))
                        inputs[col] = st.selectbox(col, options=opciones, index=default_idx)
                    else:
                        inputs[col] = st.text_input(
                            col, value="" if sample_val is None else str(sample_val)
                        )
            else:
                default_val = float(sample.get(col, 0.0) or 0.0)
                inputs[col] = st.number_input(col, value=default_val)

    submitted = st.form_submit_button("Predecir nivel de estrés", type="primary", use_container_width=True)

if submitted:
    X_input = pd.DataFrame([inputs])
    try:
        pred = model.predict(X_input)[0]
        st.divider()
        st.subheader("Resultado")
        st.success(f"Nivel predicho: **{pred}**")

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_input)[0]
            classes = getattr(model, "classes_", [f"Clase {i}" for i in range(len(proba))])
            proba_df = (
                pd.DataFrame({"Clase": classes, "Probabilidad": proba})
                .sort_values("Probabilidad", ascending=False)
                .set_index("Clase")
            )
            st.bar_chart(proba_df)
            st.dataframe(proba_df.style.format({"Probabilidad": "{:.2%}"}), use_container_width=True)
    except Exception as exc:
        st.error(f"Error al predecir: {exc}")