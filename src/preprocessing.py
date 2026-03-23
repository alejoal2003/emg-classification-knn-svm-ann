import numpy as np
from scipy.signal import butter, filtfilt

from src.config import Config

def rectify_emg(signal: np.ndarray, method: str = 'abs') -> np.ndarray:
    """
    Rectifica la señal EMG.
    Args:
        signal: array [T, C] con la señal en el tiempo.
        method: 'abs' (valor absoluto) o 'square' (cuadrado).
    Returns:
        Señal rectificada.
    """
    if method == 'square':
        return signal ** 2
    elif method == 'abs':
        return np.abs(signal)
    elif method == 'none':
        return signal
    else:
        raise ValueError("Método de rectificación inválido. Use 'abs', 'square' o 'none'.")

def bandpass_filter(signal: np.ndarray, fs: float = Config.SAMPLE_RATE, 
                    lowcut: float = Config.LOW_CUTOFF, 
                    highcut: float = Config.HIGH_CUTOFF, 
                    order: int = Config.FILTER_ORDER) -> np.ndarray:
    """
    Aplica un filtro Butterworth a la señal EMG.
    Referencia del paper: butter(5, [20 90]/(fs/2), 'bandpass') o similar.
    La guía en MATALB usa lowpass en 0.1, adaptaremos aquí un pasabanda común para sEMG.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
    # En la guía original (Shared.m), usan: butter(5, 0.1, 'low')
    # Para sEMG, comúnmente se usa pasabanda 20-500 Hz, pero Myo es 200 Hz, por lo que NYQ = 100 Hz.
    # Por mantener fidelidad a su código, usaremos pasabajos de butter(5, 0.1) si se pide.
    # Aquí implementaremos el pasabanda estándar sEMG (20-90 Hz):
    
    b, a = butter(order, [low, high], btype='bandpass')
    
    filtered_signal = np.zeros_like(signal)
    # Filtrar cada canal
    for i in range(signal.shape[1]):
        filtered_signal[:, i] = filtfilt(b, a, signal[:, i])
    return filtered_signal

def normalize_signal(signal: np.ndarray) -> np.ndarray:
    """
    Normaliza la señal. Como en Shared.m, si el valor absoluto máximo > 1, divide por 128.
    """
    max_val = np.max(np.abs(signal))
    if max_val > 1.0:
        return signal / 128.0
    return signal

def preprocess_pipeline(raw_signal: np.ndarray) -> np.ndarray:
    """
    Pipeline completo de preprocesado:
    1. Normalización (si excede 1, div 128)
    2. Rectificación Absoluta
    3. Filtrado (Butterworth pasabanda)
    """
    signal = normalize_signal(raw_signal)
    signal = rectify_emg(signal, method='abs')
    signal = bandpass_filter(signal)
    return signal

def apply_windowing(signal: np.ndarray, window_size: int = Config.WINDOW_SIZE, 
                    step: int = Config.WINDOW_STEP) -> List[np.ndarray]:
    """
    Corta la señal en ventanas (ventanas deslizantes).
    Args:
        signal: [T, C] array de señal.
        window_size: Tamaño de la ventana en muestras.
        step: Paso de la ventana en muestras.
    Returns:
        Lista de ventanas [window_size, C].
    """
    windows = []
    num_windows = (len(signal) - window_size) // step + 1
    
    for i in range(num_windows):
        start = i * step
        end = start + window_size
        window = signal[start:end, :]
        windows.append(window)
        
    return windows
