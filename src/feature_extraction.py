import numpy as np
from typing import List

# Features estadísticos y del dominio del tiempo (Time-Domain)
# Estándar en la literatura de HGR con sEMG clásico.

def compute_mav(window: np.ndarray) -> np.ndarray:
    """Mean Absolute Value (MAV)."""
    return np.mean(np.abs(window), axis=0)

def compute_rms(window: np.ndarray) -> np.ndarray:
    """Root Mean Square (RMS)."""
    return np.sqrt(np.mean(window ** 2, axis=0))

def compute_wl(window: np.ndarray) -> np.ndarray:
    """Waveform Length (WL)."""
    return np.sum(np.abs(np.diff(window, axis=0)), axis=0)

def compute_zc(window: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """Zero Crossings (ZC)."""
    # Centrar la señal primero
    window = window - np.mean(window, axis=0)
    
    diff_signs = np.diff(np.sign(window), axis=0)
    diff_vals = np.abs(np.diff(window, axis=0))
    
    zc = np.sum((diff_signs != 0) & (diff_vals >= threshold), axis=0)
    return zc

def compute_ssc(window: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """Slope Sign Changes (SSC)."""
    diff_window = np.diff(window, axis=0)
    
    diff_signs = np.diff(np.sign(diff_window), axis=0)
    diff_vals = np.abs(np.diff(diff_window, axis=0))
    
    ssc = np.sum((diff_signs != 0) & (diff_vals >= threshold), axis=0)
    return ssc

def compute_var(window: np.ndarray) -> np.ndarray:
    """Variance (VAR)."""
    return np.var(window, axis=0)

def extract_features(window: np.ndarray) -> np.ndarray:
    """
    Extrae un vector de características para una ventana de tiempo dada.
    Args:
        window: [T, C] numpy array de una ventana.
    Returns:
        [1, F] numpy array concatenando todos los features de todos los canales.
        Para 8 canales y 6 features = 48 valores.
    """
    mav = compute_mav(window)
    rms = compute_rms(window)
    wl = compute_wl(window)
    zc = compute_zc(window, threshold=0.01) # Threshold arbitrario bajo para ZC sEMG
    ssc = compute_ssc(window, threshold=0.01)
    var = compute_var(window)
    
    # Concatenar todos los features en un solo vector 1D
    feature_vector = np.concatenate([mav, rms, wl, zc, ssc, var])
    return feature_vector

def build_feature_matrix(windows: List[np.ndarray]) -> np.ndarray:
    """
    Construye la matriz X de muestras para el modelo.
    Args:
        windows: Lista de ventanas [window_size, C]
    Returns:
        X: array de shape [num_windows, num_features * C]
    """
    num_windows = len(windows)
    
    # Pre-calcular el tamaño del vector extrayendo de la primera ventana
    if num_windows == 0:
        return np.array([])
        
    first_feat = extract_features(windows[0])
    num_features = len(first_feat)
    
    X = np.zeros((num_windows, num_features))
    for i, w in enumerate(windows):
        X[i, :] = extract_features(w)
        
    return X
