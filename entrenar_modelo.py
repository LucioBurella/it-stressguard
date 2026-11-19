"""
Entrena el clasificador de IT-StressGuard.
Uso:
    python entrenar_modelo.py --csv dataset_it_stressguard.csv
"""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer


NUMERICAS = ["horasTrabajadas", "ticketsResueltos", "nivelFatiga"]
CATEGORICAS = ["rol"]
TEXTO = "comentarioDia"
OBJETIVO = "nivelEstres"
GRUPOS = "idUsuario"


def construir_pipeline():
    preprocesador = ColumnTransformer(
        transformers=[
            (
                "numericas",
                Pipeline([
                    ("imputar", SimpleImputer(strategy="median")),
                    ("escalar", StandardScaler()),
                ]),
                NUMERICAS,
            ),
            (
                "categoricas",
                Pipeline([
                    ("imputar", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                CATEGORICAS,
            ),
            (
                "texto",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    max_features=5000,
                    min_df=1,
                ),
                TEXTO,
            ),
        ],
        remainder="drop",
    )

    clasificador = RandomForestClassifier(
        n_estimators=350,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline([
        ("preprocesamiento", preprocesador),
        ("clasificador", clasificador),
    ])


def main():
    parser = argparse.ArgumentParser(description="Entrenamiento de IT-StressGuard")
    parser.add_argument("--csv", default="dataset_it_stressguard.csv",
                        help="Ruta del dataset CSV")
    parser.add_argument("--salida", default="modelo_it_stressguard.joblib",
                        help="Ruta del modelo entrenado")
    parser.add_argument("--test-size", type=float, default=0.20,
                        help="Proporción de usuarios reservada para prueba")
    args = parser.parse_args()

    ruta_csv = Path(args.csv)
    if not ruta_csv.exists():
        raise FileNotFoundError(
            f"No se encontró {ruta_csv}. Descargá el CSV y ejecutá el script en esa carpeta."
        )

    df = pd.read_csv(ruta_csv)

    requeridas = NUMERICAS + CATEGORICAS + [TEXTO, OBJETIVO, GRUPOS]
    faltantes = [c for c in requeridas if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas obligatorias en el CSV: {faltantes}")

    df = df.dropna(subset=[OBJETIVO, GRUPOS]).copy()
    X = df[NUMERICAS + CATEGORICAS + [TEXTO]].copy()
    y = df[OBJETIVO].astype(str)
    grupos = df[GRUPOS]

    # La separación se hace por usuario para evitar que registros de la misma
    # persona aparezcan simultáneamente en entrenamiento y prueba.
    separador = GroupShuffleSplit(
        n_splits=1, test_size=args.test_size, random_state=42
    )
    indices_train, indices_test = next(separador.split(X, y, groups=grupos))

    X_train, X_test = X.iloc[indices_train], X.iloc[indices_test]
    y_train, y_test = y.iloc[indices_train], y.iloc[indices_test]

    modelo = construir_pipeline()
    modelo.fit(X_train, y_train)

    predicciones = modelo.predict(X_test)
    accuracy = accuracy_score(y_test, predicciones)
    f1_macro = f1_score(y_test, predicciones, average="macro")

    print("\n=== RESULTADOS EN CONJUNTO DE PRUEBA ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 macro: {f1_macro:.4f}")
    print("\nReporte por clase:")
    print(classification_report(y_test, predicciones, zero_division=0))
    print("Matriz de confusión (orden alfabético de clases):")
    clases = sorted(y.unique())
    print(pd.DataFrame(
        confusion_matrix(y_test, predicciones, labels=clases),
        index=[f"Real {c}" for c in clases],
        columns=[f"Predicho {c}" for c in clases],
    ))

    paquete = {
        "modelo": modelo,
        "clases": list(modelo.named_steps["clasificador"].classes_),
        "metricas_prueba": {
            "accuracy": float(accuracy),
            "f1_macro": float(f1_macro),
            "cantidad_train": int(len(indices_train)),
            "cantidad_test": int(len(indices_test)),
        },
        "version": "1.0-sintetico",
    }
    joblib.dump(paquete, args.salida)

    metricas_path = Path(args.salida).with_suffix(".metricas.json")
    metricas_path.write_text(
        json.dumps(paquete["metricas_prueba"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nModelo guardado en: {args.salida}")
    print(f"Métricas guardadas en: {metricas_path}")
    print("\nIMPORTANTE: los resultados miden el dataset sintético de demostración,")
    print("no el desempeño esperado con trabajadores o datos reales.")


if __name__ == "__main__":
    main()
