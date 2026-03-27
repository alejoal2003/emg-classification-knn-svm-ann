import numpy as np
from typing import List

# ==============================
# Features del dominio del tiempo (Time-Domain)
# ==============================

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

# ==============================
# Features del dominio de la frecuencia (Frequency-Domain)
# ==============================

def compute_mnf(window: np.ndarray) -> np.ndarray:
    """Mean Frequency (MNF) — frecuencia media ponderada por la PSD."""
    T, C = window.shape
    mnf = np.zeros(C)
    for ch in range(C):
        fft_vals = np.fft.rfft(window[:, ch])
        psd = np.abs(fft_vals) ** 2
        freqs = np.fft.rfftfreq(T)
        total_power = np.sum(psd)
        if total_power > 0:
            mnf[ch] = np.sum(freqs * psd) / total_power
        else:
            mnf[ch] = 0.0
    return mnf

def compute_mdf(window: np.ndarray) -> np.ndarray:
    """Median Frequency (MDF) — frecuencia que divide la PSD en dos mitades iguales."""
    T, C = window.shape
    mdf = np.zeros(C)
    for ch in range(C):
        fft_vals = np.fft.rfft(window[:, ch])
        psd = np.abs(fft_vals) ** 2
        freqs = np.fft.rfftfreq(T)
        cumulative = np.cumsum(psd)
        total_power = cumulative[-1]
        if total_power > 0:
            idx = np.searchsorted(cumulative, total_power / 2.0)
            idx = min(idx, len(freqs) - 1)
            mdf[ch] = freqs[idx]
        else:
            mdf[ch] = 0.0
    return mdf

def compute_se(window: np.ndarray) -> np.ndarray:
    """Spectral Energy (SE) — energía total del espectro."""
    T, C = window.shape
    se = np.zeros(C)
    for ch in range(C):
        fft_vals = np.fft.rfft(window[:, ch])
        se[ch] = np.sum(np.abs(fft_vals) ** 2)
    return se

# ==============================
# Extracción unificada
# ==============================

def extract_features(window: np.ndarray) -> np.ndarray:
    """
    Extrae un vector de características para una ventana de tiempo dada.
    Incluye 6 features de tiempo + 3 features de frecuencia = 9 features por canal.
    Para 8 canales → 72 valores.
    Args:
        window: [T, C] numpy array de una ventana.
    Returns:
        [F,] numpy array concatenando todos los features de todos los canales.
    """
    # Time-domain
    mav = compute_mav(window)
    rms = compute_rms(window)
    wl  = compute_wl(window)
    zc  = compute_zc(window, threshold=0.01)
    ssc = compute_ssc(window, threshold=0.01)
    var = compute_var(window)
    # Frequency-domain
    mnf = compute_mnf(window)
    mdf = compute_mdf(window)
    se  = compute_se(window)

    feature_vector = np.concatenate([mav, rms, wl, zc, ssc, var, mnf, mdf, se])
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
    if num_windows == 0:
        return np.array([])

    first_feat = extract_features(windows[0])
    num_features = len(first_feat)

    X = np.zeros((num_windows, num_features))
    for i, w in enumerate(windows):
        X[i, :] = extract_features(w)

    return X

