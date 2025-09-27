# clasificacion.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier  # 🔹 Aquí puedes cambiar el algoritmo
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import os

# =========================
# Configuración y paths
# =========================
DATA_PATH = './Datasheets/data.csv'
CONF_IMG_PATH = 'static/clf_confusion_matrix.png'

# =========================
# Carga de datos
# =========================
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Si tienes variables categóricas, aplica map o get_dummies aquí.
    if 'Experiencia' in df.columns and df['Experiencia'].dtype == 'object':
        exp_map = {'Baja': 0, 'Media': 1, 'Alta': 2}
        df['Experiencia'] = df['Experiencia'].map(exp_map).astype(int)

    return df

# =========================
# Split
# =========================
def split_xy(df: pd.DataFrame):
    X = df.drop('Fracaso', axis=1)
    y = df['Fracaso']
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# =========================
# Entrenamiento
# =========================
def train_model(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Usa un Pipeline para evitar leakage (escalado + modelo).
    Cambia el clasificador aquí según el algoritmo que necesites.
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=5))  # 🔹 cambia por SVM, MLP, etc.
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

def _plot_confusion(cm: np.ndarray, save_path: str = CONF_IMG_PATH):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(6,5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', cbar=False,
        xticklabels=['No (0)', 'Sí (1)'], yticklabels=['No (0)', 'Sí (1)']
    )
    plt.xlabel('Predicho')
    plt.ylabel('Real')
    plt.title('Matriz de Confusión — Algoritmo de Clasificación')
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()

# =========================
# Evaluación
# =========================
def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    rep_dict = classification_report(
        y_test, y_pred, target_names=['No', 'Sí'], output_dict=True
    )
    rep_df = pd.DataFrame(rep_dict).transpose().round(3)
    report_html = rep_df.to_html(classes='table table-striped table-sm', border=0)

    cm = confusion_matrix(y_test, y_pred)
    _plot_confusion(cm, CONF_IMG_PATH)

    return round(acc, 4), report_html, cm, CONF_IMG_PATH

# =========================
# Predicción individual
# =========================
def predict_label(model, features: dict, threshold: float = 0.5):
    """
    features (dict):
      {
        'CapitalInicial': float,
        'Experiencia': 0|1|2,
        'NumSocios': int,
        'AniosOperacion': int
      }
    """
    row = np.array([
        float(features['CapitalInicial']),
        int(features['Experiencia']),
        float(features['NumSocios']),
        float(features['AniosOperacion'])
    ]).reshape(1, -1)

    prob = model.predict_proba(row)[0, 1] if hasattr(model, "predict_proba") else model.decision_function(row)
    label = 'Sí' if prob >= float(threshold) else 'No'
    return label, float(prob)

# =========================
# Informe breve
# =========================
def informe_breve() -> str:
    return ("Flujo: carga (CSV) → split (80/20) → preprocesamiento (Pipeline con escalado) → "
            "entrenamiento (algoritmo de clasificación elegido) → evaluación (accuracy, reporte, matriz de confusión) → "
            "predicción (Sí/No y probabilidad).")

# =========================
# Ejecución directa
# =========================
if __name__ == '__main__':
    data = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = split_xy(data)
    model = train_model(X_train, y_train)
    accuracy, report_html, cm, img_path = evaluate(model, X_test, y_test)

    print(f'Exactitud del modelo: {accuracy * 100:.2f}%')
    print("Matriz de confusión:\n", cm)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=['No', 'Sí']))
