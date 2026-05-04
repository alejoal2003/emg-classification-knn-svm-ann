"""
activity_detection.py — Detección de Onset/Offset de actividad muscular en señales sEMG

El "reconocimiento" complementa a la "clasificación":
  - Clasificación: ¿QUÉ gesto se está ejecutando?
  - Reconocimiento: ¿CUÁNDO inicia y CUÁNDO termina el movimiento?

Método: Umbralización dinámica de la envolvente RMS de la señal EMG.
  1. Se calcula la energía RMS con ventana deslizante.
  2. Se estima el ruido base (baseline) a partir de las primeras muestras o de un segmento de reposo.
  3. Se fija un umbral = media_ruido + k * std_ruido (default k=3).
  4. Se detectan los cruces por encima (Onset) y por debajo (Offset) del umbral.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List, Optional


def calculate_moving_rms(signal: np.ndarray, window_size: int = 40) -> np.ndarray:
    """
    Calcula la energía RMS usando una ventana deslizante, promediada entre canales.

    Args:
        signal: array (T, C) — T muestras temporales, C canales EMG.
        window_size: tamaño de la ventana en muestras (ej. 200 Hz * 0.2s = 40).

    Returns:
        rms_energy: array (T,) — energía RMS promedio entre canales.
    """
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)

    n_samples, n_channels = signal.shape
    squared = signal.astype(np.float64) ** 2

    # Usar cumsum para ventana deslizante eficiente
    cumsum = np.vstack([np.zeros((1, n_channels)), np.cumsum(squared, axis=0)])
    half_win = window_size // 2

    rms = np.zeros((n_samples, n_channels))
    for i in range(n_samples):
        start = max(0, i - half_win)
        end = min(n_samples, i + half_win + 1)
        rms[i] = np.sqrt(np.mean(squared[start:end], axis=0))

    return np.mean(rms, axis=1)


def estimate_noise_threshold(signal: np.ndarray,
                             window_size: int = 40,
                             noise_samples: Optional[int] = None,
                             multiplier: float = 3.0
                             ) -> Tuple[float, float, float]:
    """
    Estima el umbral de detección a partir del ruido de fondo.

    Args:
        signal: array (T, C) — señal EMG (puede ser solo la porción de reposo).
        window_size: tamaño de ventana para RMS.
        noise_samples: muestras iniciales a usar como baseline (None = todas).
        multiplier: k en la fórmula umbral = μ + k·σ.

    Returns:
        threshold, noise_mean, noise_std
    """
    rms = calculate_moving_rms(signal, window_size)

    if noise_samples is not None:
        rms = rms[:noise_samples]

    noise_mean = float(np.mean(rms))
    noise_std = float(np.std(rms))
    threshold = noise_mean + multiplier * noise_std

    return threshold, noise_mean, noise_std


def detect_onset_offset(energy: np.ndarray,
                        threshold: float,
                        min_duration: int = 20
                        ) -> List[Tuple[int, int]]:
    """
    Detecta segmentos de actividad muscular (Onset → Offset).

    Args:
        energy: array (T,) — energía RMS de la señal.
        threshold: umbral de detección.
        min_duration: duración mínima en muestras para validar un segmento.

    Returns:
        segments: lista de tuplas (onset_idx, offset_idx).
    """
    active = energy > threshold
    segments = []
    in_segment = False
    onset = 0

    for i in range(len(active)):
        if active[i] and not in_segment:
            onset = i
            in_segment = True
        elif not active[i] and in_segment:
            if (i - onset) >= min_duration:
                segments.append((onset, i))
            in_segment = False

    # Si la señal termina mientras aún hay actividad
    if in_segment and (len(active) - onset) >= min_duration:
        segments.append((onset, len(active) - 1))

    return segments


def plot_onset_offset(signal: np.ndarray,
                      gesture_name: str = "Gesto",
                      window_size: int = 40,
                      noise_samples: int = 100,
                      multiplier: float = 3.0,
                      sample_rate: int = 200,
                      save_path: Optional[Path] = None) -> List[Tuple[int, int]]:
    """
    Visualiza la detección de Onset/Offset sobre una señal EMG cruda.

    Genera una figura de 3 paneles:
      1. Señal EMG cruda (todos los canales).
      2. Envolvente RMS + línea de umbral + marcadores Onset/Offset.
      3. Segmentos activos resaltados sobre la señal original.

    Args:
        signal: array (T, C) — señal EMG cruda de una repetición.
        gesture_name: nombre del gesto (para el título).
        window_size: ventana RMS en muestras.
        noise_samples: muestras iniciales para estimar ruido base.
        multiplier: multiplicador de σ para el umbral.
        sample_rate: frecuencia de muestreo (Hz).
        save_path: ruta para guardar la figura (None = solo mostrar).

    Returns:
        segments: lista de tuplas (onset_idx, offset_idx) detectados.
    """
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)

    energy = calculate_moving_rms(signal, window_size)
    threshold, noise_mean, noise_std = estimate_noise_threshold(
        signal, window_size, noise_samples, multiplier
    )
    segments = detect_onset_offset(energy, threshold, min_duration=20)
    time = np.arange(signal.shape[0]) / sample_rate

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # --- Panel 1: Señal EMG cruda ---
    ax1 = axes[0]
    for ch in range(signal.shape[1]):
        ax1.plot(time, signal[:, ch], alpha=0.5, linewidth=0.5)
    ax1.set_ylabel('Amplitud EMG')
    ax1.set_title(f'Señal EMG Cruda — {gesture_name}', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: Envolvente RMS + Umbral ---
    ax2 = axes[1]
    ax2.plot(time, energy, color='#1f77b4', linewidth=1.5, label='Energía RMS')
    ax2.axhline(y=threshold, color='red', linestyle='--', linewidth=1.5,
                label=f'Umbral (μ+{multiplier}σ = {threshold:.2f})')
    ax2.axhline(y=noise_mean, color='gray', linestyle=':', linewidth=1,
                label=f'Media ruido = {noise_mean:.2f}')

    for onset, offset in segments:
        t_on = onset / sample_rate
        t_off = offset / sample_rate
        ax2.axvline(x=t_on, color='green', linestyle='-', linewidth=2, alpha=0.8)
        ax2.axvline(x=t_off, color='orange', linestyle='-', linewidth=2, alpha=0.8)
        ax2.annotate('ONSET', xy=(t_on, threshold),
                     xytext=(t_on + 0.05, threshold * 1.3),
                     fontsize=9, fontweight='bold', color='green',
                     arrowprops=dict(arrowstyle='->', color='green'))
        ax2.annotate('OFFSET', xy=(t_off, threshold),
                     xytext=(t_off + 0.05, threshold * 1.3),
                     fontsize=9, fontweight='bold', color='orange',
                     arrowprops=dict(arrowstyle='->', color='orange'))

    ax2.set_ylabel('Energía RMS')
    ax2.set_title('Envolvente de Energía + Umbral Dinámico', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # --- Panel 3: Segmentos activos resaltados ---
    ax3 = axes[2]
    for ch in range(signal.shape[1]):
        ax3.plot(time, signal[:, ch], alpha=0.3, linewidth=0.5, color='gray')

    for idx, (onset, offset) in enumerate(segments):
        t_on = onset / sample_rate
        t_off = offset / sample_rate
        lbl = 'Segmento activo' if idx == 0 else None
        ax3.axvspan(t_on, t_off, alpha=0.25, color='green', label=lbl)
        for ch in range(signal.shape[1]):
            ax3.plot(time[onset:offset], signal[onset:offset, ch],
                     alpha=0.6, linewidth=0.8)

    ax3.set_ylabel('Amplitud EMG')
    ax3.set_xlabel('Tiempo (s)')
    ax3.set_title('Segmentos de Actividad Detectados', fontsize=13, fontweight='bold')
    if segments:
        total_ms = sum((off - on) for on, off in segments) / sample_rate * 1000
        ax3.legend([f'{len(segments)} segmento(s), {total_ms:.0f} ms total'],
                   loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Gráfica guardada: {save_path}')
    plt.show()

    # Resumen textual
    print(f'\n--- Resumen de Detección: {gesture_name} ---')
    print(f'Umbral: {threshold:.4f}  (μ={noise_mean:.4f}, σ={noise_std:.4f}, k={multiplier})')
    print(f'Segmentos detectados: {len(segments)}')
    for i, (on, off) in enumerate(segments):
        dur = (off - on) / sample_rate * 1000
        print(f'  #{i+1}: Onset={on/sample_rate:.3f}s  Offset={off/sample_rate:.3f}s  '
              f'Duración={dur:.0f} ms')

    return segments
