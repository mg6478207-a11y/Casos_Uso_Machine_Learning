# regresionLogistica.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import os

# =========================
# Configuración y mapeos
# =========================
DATA_PATH = './Datasheets/data.csv'
CONF_IMG_PATH = 'static/rl_confusion_matrix.png'

# Mapeo ordinal para la variable categórica Experiencia
# Baja=0, Media=1, Alta=2
EXPERIENCIA_MAP = {'Baja': 0, 'Media': 1, 'Alta': 2}

# =========================
# Carga y exploración
# =========================
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Carga dataset de predicción de fracaso en emprendimientos.

    Variables independientes:
        - CapitalInicial (numérica)
        - Experiencia (categórica: Baja=0, Media=1, Alta=2)
        - NumSocios (numérica)
        - AniosOperacion (numérica)
    Variable objetivo:
        - Fracaso (1=Sí, 0=No). Clase positiva: 1 (“Sí”), negativa: 0 (“No”).
    """
    df = pd.read_csv(path)

    # Codificar 'Experiencia' a ordinal si viene como texto
    if 'Experiencia' in df.columns and df['Experiencia'].dtype == 'object':
        df['Experiencia'] = df['Experiencia'].map(EXPERIENCIA_MAP)

    # Asegurar tipo entero
    if 'Experiencia' in df.columns:
        df['Experiencia'] = df['Experiencia'].astype(int)

    # Exploración básica (estilo ejemplo del docente)
    print(df.head())
    print(df.info())
    print(df.describe())

    return df

# =========================
# Separación X / y y split
# =========================
def split_xy(df: pd.DataFrame):
    x = df.drop('Fracaso', axis=1)
    y = df['Fracaso']
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test

# =========================
# Entrenamiento (con escalado)
# =========================
def train_model(X_train: pd.DataFrame, y_train: pd.Series):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    logistic_model = LogisticRegression()  # regularización por defecto (L2)
    logistic_model.fit(X_train_scaled, y_train)

    return logistic_model, scaler

def _plot_confusion(cm: np.ndarray, save_path: str = CONF_IMG_PATH):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(6,5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', cbar=False,
        xticklabels=['No (0)', 'Sí (1)'], yticklabels=['No (0)', 'Sí (1)']
    )
    plt.xlabel('Predicho')
    plt.ylabel('Real')
    plt.title('Matriz de Confusión — Fracaso Emprendimiento')
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()

# =========================
# Evaluación
# =========================
def evaluate(model, scaler, X_test: pd.DataFrame, y_test: pd.Series):
    """
    Retorna:
      - accuracy (float, 2–4 decimales)
      - report_html (tabla HTML con precision, recall, f1-score, support)
      - cm (np.ndarray 2x2)
      - img_path (ruta de la imagen generada)
    """
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)

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
def experiencia_str_to_code(exp_str: str) -> int:
    return EXPERIENCIA_MAP.get(exp_str, 1)  # por defecto Media=1

def predict_label(model, scaler, features: dict, threshold: float = 0.5):
    """
    features (dict):
      {
        'CapitalInicial': float,
        'Experiencia': 'Baja'|'Media'|'Alta' o 0|1|2,
        'NumSocios': float/int,
        'AniosOperacion': float/int
      }
    Retorna: ('Sí'/'No', prob_float)
    """
    exp_val = features.get('Experiencia')
    if isinstance(exp_val, str):
        exp_code = experiencia_str_to_code(exp_val)
    else:
        exp_code = int(exp_val)

    row = np.array([
        float(features['CapitalInicial']),
        float(exp_code),
        float(features['NumSocios']),
        float(features['AniosOperacion'])
    ]).reshape(1, -1)

    row_scaled = scaler.transform(row)
    prob = model.predict_proba(row_scaled)[0, 1]
    label = 'Sí' if prob >= float(threshold) else 'No'
    return label, float(prob)

# =========================
# Informe breve del flujo
# =========================
def informe_breve() -> str:
    return ("Flujo: carga (CSV) → split (80/20, estratificado) → estandarización (StandardScaler) → "
            "entrenamiento (LogisticRegression) → evaluación (accuracy, reporte, matriz de confusión) → "
            "predicción (etiqueta Sí/No y probabilidad).")

# =========================
# Ejecución directa (estilo docente)
# =========================
if __name__ == '__main__':
    data = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = split_xy(data)
    logistic_model, scaler = train_model(X_train, y_train)
    accuracy, report_html, cm, img_path = evaluate(logistic_model, scaler, X_test, y_test)

    print(f'Exactitud del modelo: {accuracy * 100:.2f}%')
    print("Matriz de confusión:\n", cm)
    y_pred = logistic_model.predict(scaler.transform(X_test))
    print(classification_report(y_test, y_pred, target_names=['No', 'Sí']))
