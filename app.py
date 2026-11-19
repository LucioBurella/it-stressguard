from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="IT-StressGuard — Panel Analítico",
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

PERFILES_EJEMPLO = {
    "Jornada equilibrada (Dev)": {
        "horas": 7.5,
        "tickets": 5,
        "fatiga": 3,
        "rol": "Dev",
        "comentario": "Jornada ordenada, buen avance en tareas planificadas y pausas respetadas.",
    },
    "Carga exigente / guardias (DevOps)": {
        "horas": 10.5,
        "tickets": 14,
        "fatiga": 8,
        "rol": "DevOps",
        "comentario": "Varios incidentes en producción y despliegues urgentes durante la tarde.",
    },
    "Alta presión de entregas (Team Lead)": {
        "horas": 11.0,
        "tickets": 9,
        "fatiga": 9,
        "rol": "Team Lead",
        "comentario": "Muchas reuniones, bloqueos de equipo y cierres ajustados sin descanso.",
    },
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


def detect_target_col(df: pd.DataFrame | None) -> str | None:
    if df is None:
        return None
    candidates = [
        c for c in df.columns
        if c.lower() in {"nivelestres", "nivel_estres", "estres", "stress", "target", "label"}
    ]
    return candidates[0] if candidates else None


artifact = load_artifact(MODEL_PATH)
model = get_model(artifact)
df = load_dataset(DATA_PATH)
metrics_file = load_metrics(METRICS_PATH)

# --- Sidebar ---
st.sidebar.title("IT-StressGuard")
st.sidebar.caption("Sistema de apoyo para estimación y análisis de estrés laboral en IT")

menu = st.sidebar.radio(
    "Secciones",
    [
        "Predicción y Diagnóstico",
        "Exploración del Dataset",
        "Modelo y Métricas",
        "Arquitectura y Uso",
    ],
)

st.sidebar.divider()
st.sidebar.markdown("**Estado de archivos:**")
st.sidebar.write(f"- Modelo: {'Disponible' if model is not None else 'No encontrado'}")
st.sidebar.write(f"- Dataset: {'Disponible' if df is not None else 'No encontrado'}")
st.sidebar.write(f"- Métricas: {'Disponible' if metrics_file is not None else 'No encontrado'}")
st.sidebar.caption("Prototipo académico con datos sintéticos; no es diagnóstico médico.")

# ==============================================================================
# 1. PREDICCIÓN Y DIAGNÓSTICO
# ==============================================================================
if menu == "Predicción y Diagnóstico":
    st.title("Predicción y Diagnóstico de Estrés Laboral")
    st.write(
        "Ingresá los parámetros de la jornada o cargá un perfil de ejemplo para simular "
        "la estimación del modelo (`RandomForestClassifier + TF-IDF`)."
    )

    if model is None:
        st.error("No se pudo cargar `modelo_it_stressguard.joblib` o falta la clave `'modelo'`.")
        st.stop()

    perfil_sel = st.selectbox(
        "Cargar escenario predefinido (opcional)",
        options=["Personalizado"] + list(PERFILES_EJEMPLO.keys()),
    )
    preset = PERFILES_EJEMPLO.get(perfil_sel, {})

    with st.form("form_prediccion_detallada"):
        c1, c2, c3 = st.columns(3)
        with c1:
            horas = st.slider(
                "Horas trabajadas",
                min_value=0.0,
                max_value=24.0,
                value=float(preset.get("horas", 8.0)),
                step=0.5,
                help="Total de horas dedicadas a la jornada laboral.",
            )
        with c2:
            tickets = st.number_input(
                "Tickets resueltos",
                min_value=0,
                max_value=500,
                value=int(preset.get("tickets", 6)),
                step=1,
                help="Cantidad de incidencias, tareas o tickets cerrados.",
            )
        with c3:
            fatiga = st.slider(
                "Fatiga percibida (1 a 10)",
                min_value=1,
                max_value=10,
                value=int(preset.get("fatiga", 5)),
                step=1,
                help="1 = sin fatiga / descansado; 10 = agotamiento extremo.",
            )

        roles_opts = ["Dev", "DevOps", "QA", "Team Lead", "HR"]
        default_rol = preset.get("rol", "Dev")
        rol_idx = roles_opts.index(default_rol) if default_rol in roles_opts else 0

        c4, c5 = st.columns([1, 2])
        with c4:
            rol = st.selectbox("Rol en el equipo", options=roles_opts, index=rol_idx)
        with c5:
            comentario_inicial = preset.get("comentario")
            comentario = st.text_input(
                "Comentario del día",
                value="" if comentario_inicial is None else str(comentario_inicial),
                placeholder="Describí brevemente cómo estuvo la jornada, bloqueos, guardias o ritmo de trabajo...",
            )

        submitted = st.form_submit_button("Ejecutar predicción", type="primary", use_container_width=True)

    if submitted:
        comentario_limpio = str(comentario or "")[:500]
        entrada = pd.DataFrame(
            [
                {
                    "horasTrabajadas": horas,
                    "ticketsResueltos": tickets,
                    "nivelFatiga": fatiga,
                    "comentarioDia": comentario_limpio,
                    "rol": rol,
                }
            ]
        )

        try:
            nivel = str(model.predict(entrada)[0])
            probabilidades = model.predict_proba(entrada)[0] if hasattr(model, "predict_proba") else None
        except Exception as exc:
            st.error(f"Ocurrió un error al predecir: {exc}")
            st.stop()

        st.divider()
        st.subheader("Diagnóstico generado")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Nivel estimado", nivel)
        if probabilidades is not None:
            sorted_probs = sorted([float(p) for p in probabilidades], reverse=True)
            prob_max = sorted_probs[0]
            margen = (sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else sorted_probs[0]
            m2.metric("Probabilidad principal", f"{prob_max:.1%}")
            m3.metric("Margen vs. 2.ª clase", f"{margen:.1%}")
        m4.metric("Relación tickets / hora", f"{(tickets / horas):.2f}" if horas > 0 else "0.00")

        st.success(f"**Acción sugerida:** {RECOMENDACIONES.get(nivel, 'Revisar el resultado con un profesional.')}")

        alertas = []
        if horas >= 10:
            alertas.append("Jornada extensa (>= 10 horas).")
        if fatiga >= 8:
            alertas.append("Nivel de fatiga elevado (>= 8/10).")
        if not comentario_limpio.strip():
            alertas.append("No se ingresó comentario textual (el componente TF-IDF recibe texto vacío).")
        if alertas:
            st.warning("**Observaciones del registro:** " + " ".join(alertas))

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
                st.markdown("**Distribución de probabilidades por clase**")
                st.bar_chart(prob_df.set_index("Clase")["Probabilidad"])
            with g2:
                st.markdown("**Detalle porcentual**")
                st.dataframe(
                    prob_df.style.format({"Probabilidad": "{:.2%}"}),
                    use_container_width=True,
                    hide_index=True,
                )

        with st.expander("Ver payload estructurado enviado al modelo"):
            st.dataframe(entrada, use_container_width=True, hide_index=True)

# ==============================================================================
# 2. EXPLORACIÓN DEL DATASET
# ==============================================================================
elif menu == "Exploración del Dataset":
    st.title("Exploración del Dataset")

    if df is None or df.empty:
        st.warning("No se encontró `dataset_it_stressguard.csv` o está vacío.")
        st.stop()

    target_col = detect_target_col(df)
    role_col = "rol" if "rol" in df.columns else ("Rol" if "Rol" in df.columns else None)

    f1, f2 = st.columns(2)
    df_filtrado = df.copy()

    if role_col:
        roles_unicos = ["Todos"] + sorted(df[role_col].dropna().astype(str).unique().tolist())
        sel_rol = f1.selectbox("Filtrar por rol", options=roles_unicos)
        if sel_rol != "Todos":
            df_filtrado = df_filtrado[df_filtrado[role_col].astype(str) == sel_rol]

    if target_col:
        clases_unicas = ["Todas"] + sorted(df[target_col].dropna().astype(str).unique().tolist())
        sel_clase = f2.selectbox(f"Filtrar por {target_col}", options=clases_unicas)
        if sel_clase != "Todas":
            df_filtrado = df_filtrado[df_filtrado[target_col].astype(str) == sel_clase]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Filas visibles", len(df_filtrado), delta=f"de {len(df)} total")
    k2.metric("Columnas", len(df_filtrado.columns))
    k3.metric("Celdas nulas", int(df_filtrado.isna().sum().sum()))
    k4.metric("Filas duplicadas", int(df_filtrado.duplicated().sum()))

    st.divider()

    g_left, g_right = st.columns(2)
    with g_left:
        if target_col and not df_filtrado.empty:
            st.subheader(f"Distribución de `{target_col}`")
            st.bar_chart(df_filtrado[target_col].astype(str).value_counts())
        elif role_col and not df_filtrado.empty:
            st.subheader(f"Registros por `{role_col}`")
            st.bar_chart(df_filtrado[role_col].astype(str).value_counts())

    with g_right:
        num_cols = df_filtrado.select_dtypes(include="number").columns.tolist()
        if num_cols and not df_filtrado.empty:
            num_elegida = st.selectbox("Serie numérica", options=num_cols)
            st.subheader(f"Evolución / secuencia de `{num_elegida}`")
            st.line_chart(df_filtrado[num_elegida].dropna().reset_index(drop=True))

    num_cols = df_filtrado.select_dtypes(include="number").columns.tolist()
    group_candidates = [c for c in [target_col, role_col] if c is not None]
    if group_candidates and num_cols and not df_filtrado.empty:
        st.divider()
        st.subheader("Promedios por categoría")
        grp_col = st.selectbox("Agrupar promedios numéricos por", options=group_candidates)
        resumen_grp = df_filtrado.groupby(grp_col, dropna=False)[num_cols].mean(numeric_only=True).round(2)
        st.dataframe(resumen_grp, use_container_width=True)

        metric_to_plot = st.selectbox("Graficar promedio de variable", options=num_cols, key="grp_plot")
        st.bar_chart(resumen_grp[metric_to_plot])

    st.divider()
    st.subheader("Estadísticas descriptivas (variables numéricas)")
    if num_cols and not df_filtrado.empty:
        st.dataframe(df_filtrado[num_cols].describe().T.round(2), use_container_width=True)

    st.subheader("Muestra de registros")
    st.dataframe(df_filtrado.head(100), use_container_width=True)

# ==============================================================================
# 3. MODELO Y MÉTRICAS
# ==============================================================================
elif menu == "Modelo y Métricas":
    st.title("Modelo y Métricas de Evaluación")

    version = artifact.get("version", "desconocida") if artifact else "no disponible"
    algoritmo = artifact.get("algoritmo", "RandomForestClassifier + TF-IDF") if artifact else "no disponible"
    clases = get_classes(model, artifact)

    m1, m2, m3 = st.columns(3)
    m1.metric("Versión", str(version))
    m2.metric("Pipeline / Algoritmo", str(algoritmo))
    m3.metric("Clases objetivo", len(clases))

    st.markdown("**Clases soportadas:** " + " · ".join(f"`{c}`" for c in clases))

    st.divider()

    metricas_dict = {}
    if isinstance(metrics_file, dict):
        metricas_dict.update(metrics_file)
    if artifact and isinstance(artifact.get("metricas_prueba"), dict):
        for k, v in artifact["metricas_prueba"].items():
            metricas_dict.setdefault(k, v)

    if metricas_dict:
        st.subheader("Métricas registradas")
        cols = st.columns(min(len(metricas_dict), 4))
        for idx, (k, v) in enumerate(metricas_dict.items()):
            label = k.replace("_", " ").capitalize()
            val_fmt = f"{v:.4f}" if isinstance(v, float) else str(v)
            cols[idx % len(cols)].metric(label=label, value=val_fmt)

        scale_0_1 = {
            k: float(v)
            for k, v in metricas_dict.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool) and 0.0 <= float(v) <= 1.0
        }
        if scale_0_1:
            st.markdown("**Comparativa de métricas en escala 0–1**")
            st.bar_chart(pd.Series(scale_0_1, name="Puntaje"))

        with st.expander("Ver JSON completo de métricas"):
            st.json(metricas_dict)
    else:
        st.info("No se encontraron métricas en `modelo_it_stressguard.metricas.json` ni en el `.joblib`.")

# ==============================================================================
# 4. ARQUITECTURA Y USO
# ==============================================================================
elif menu == "Arquitectura y Uso":
    st.title("Arquitectura del Prototipo y Uso")
    st.markdown(
        """
        Este proyecto reúne tres componentes principales dentro del repositorio:

        - **`entrenar_modelo.py`**: entrena el modelo sobre `dataset_it_stressguard.csv`, calcula métricas y exporta el paquete en `modelo_it_stressguard.joblib` junto con `modelo_it_stressguard.metricas.json`.
        - **`api.py`**: expone una API REST con **FastAPI** (`/health`, `/model/info` y `POST /predict`) validando entradas con Pydantic.
        - **`app.py`**: interfaz interactiva en **Streamlit** para simular predicciones, explorar el dataset y revisar métricas.
        """
    )
    st.info(
        "Para correr la API en local: `uvicorn api:app --reload` y luego abrir `http://127.0.0.1:8000/docs`."
    )