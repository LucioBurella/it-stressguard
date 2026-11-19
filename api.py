"""
API REST de inferencia para IT-StressGuard.
Arranque:
    uvicorn api:app --reload
Documentación interactiva:
    http://127.0.0.1:8000/docs
"""

from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


RUTA_MODELO = Path("modelo_it_stressguard.joblib")
app = FastAPI(
    title="IT-StressGuard API",
    description="Servicio de predicción de nivel de estrés laboral (prototipo académico).",
    version="1.0.0",
)

paquete = None


def cargar_modelo():
    global paquete
    if not RUTA_MODELO.exists():
        paquete = None
        return
    paquete = joblib.load(RUTA_MODELO)


@app.on_event("startup")
def startup():
    cargar_modelo()


class EntradaPrediccion(BaseModel):
    horas: float = Field(ge=0, le=24, description="Horas trabajadas")
    tickets: int = Field(ge=0, le=500, description="Tickets resueltos")
    fatiga: int = Field(ge=1, le=10, description="Fatiga percibida de 1 a 10")
    comentario: str = Field(default="", max_length=500)
    rol: Literal["Dev", "DevOps", "QA", "Team Lead", "HR"] = "Dev"


@app.get("/health")
def health():
    return {
        "estado": "ok" if paquete is not None else "modelo_no_cargado",
        "modeloDisponible": paquete is not None,
    }


@app.get("/model/info")
def model_info():
    if paquete is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible. Entrenalo primero.")
    return {
        "version": paquete.get("version", "desconocida"),
        "clases": paquete.get("clases", []),
        "metricas": paquete.get("metricas_prueba", {}),
        "algoritmo": "RandomForestClassifier + TF-IDF",
    }


@app.post("/predict")
def predict(datos: EntradaPrediccion):
    if paquete is None:
        raise HTTPException(
            status_code=503,
            detail="El modelo no está cargado. Ejecutá entrenar_modelo.py primero.",
        )

    entrada = pd.DataFrame([{
        "horasTrabajadas": datos.horas,
        "ticketsResueltos": datos.tickets,
        "nivelFatiga": datos.fatiga,
        "comentarioDia": datos.comentario,
        "rol": datos.rol,
    }])

    modelo = paquete["modelo"]
    nivel = str(modelo.predict(entrada)[0])

    probabilidades = modelo.predict_proba(entrada)[0]
    clases = modelo.named_steps["clasificador"].classes_
    probabilidad = float(max(probabilidades))

    recomendaciones = {
        "Bajo": "Mantener hábitos saludables, pausas regulares y organización de tareas.",
        "Medio": "Revisar prioridades, distribuir la carga y programar pausas durante la jornada.",
        "Alto": "Conversar con el líder sobre la carga de trabajo y evaluar una redistribución de tareas.",
        "Burnout Inminente": "Priorizar el descanso y solicitar acompañamiento del equipo responsable de bienestar.",
    }

    # Las clases y probabilidades corresponden al clasificador entrenado.
    # Los factores individuales tipo SHAP no se calculan en esta versión.
    return {
        "nivelEstres": nivel,
        "probabilidad": round(probabilidad, 4),
        "probabilidadesPorClase": {
            str(clase): round(float(p), 4)
            for clase, p in zip(clases, probabilidades)
        },
        "factores": [],
        "recomendacion": recomendaciones.get(nivel, "Revisar el resultado con un profesional."),
        "origen": "IA",
        "nota": "Prototipo académico con datos sintéticos; no es una herramienta de diagnóstico médico.",
    }
