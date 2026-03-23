import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from typing import Dict, List, Any

# ==========================================
# Métricas Básicas
# ==========================================
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calcula las métricas principales de clasificación multiclase.
    """
    acc = accuracy_score(y_true, y_pred)
    # macro: calcula métricas por clase y promedia
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    return {
        'accuracy': float(acc),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1)
    }

def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Retorna la matriz de confusión cruda."""
    return confusion_matrix(y_true, y_pred)

# ==========================================
# Evaluación Longitudinal
# ==========================================
def evaluate_model_on_session(model, X_session: np.ndarray, y_session: np.ndarray, is_keras: bool = False) -> Dict[str, float]:
    """
    Evalúa un modelo ya entrenado (sklearn o keras) sobre un conjunto de datos validación/test.
    """
    if is_keras:
        # Keras retorna probabilidades (N, clases)
        y_probs = model.predict(X_session)
        y_pred = np.argmax(y_probs, axis=1)
    else:
        y_pred = model.predict(X_session)
        
    metrics = compute_metrics(y_session, y_pred)
    return metrics

def build_longitudinal_results_df(results_list: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Recibe una lista de diccionarios con resultados de cada sesión y retorna un DataFrame fácil de usar para gráficas y estadística.
    Ejemplo de diccionario:
       {'model': 'kNN', 'session': 0, 'accuracy': 0.85, 'precision': 0.84, ...}
    """
    return pd.DataFrame(results_list)

# ==========================================
# Pruebas Estadísticas
# ==========================================
def run_friedman_test(df_results: pd.DataFrame, metric: str = 'accuracy'):
    """
    Ejecuta el test no paramétrico de Friedman.
    Es útil para comparar el rendimiento de diferentes modelos a lo largo del tiempo o el mismo modelo a lo largo del tiempo intra-sujetos.
    Requiere un formato ancho (pivot).
    """
    from scipy.stats import friedmanchisquare
    
    # Asumiendo que df_results tiene ['model', 'session', 'accuracy', 'user_id']
    # y queremos saber si el accuracy de un modelo (ej. kNN) cambia en el tiempo (sesiones).
    
    # OJO: La formulación exacta del test en la tesis depende de cómo extraigas los accuracy.
    # Si mides la precisión global del modelo por sesión, tendrás solo N_sesiones puntos de datos, insuficiente para Friedman.
    # Debes medir la precisión POR USUARIO y POR SESIÓN.
    
    # Ejemplo genérico (reemplazar con datos reales pivotados):
    # stat, p = friedmanchisquare(grupo1, grupo2, grupo3)
    # return p
    pass
