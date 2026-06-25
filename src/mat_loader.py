"""
mat_loader.py — Cargador de datos longitudinales en formato .mat
"""

import numpy as np
import scipy.io
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import warnings

# Gestos que se clasifican (incluye relax)
GESTURES = ['fist', 'open', 'pinch', 'waveIn', 'waveOut', 'relax']

def _load_gesture_mat(mat_path: Path, gesture_name: str) -> Optional[np.ndarray]:
    """
    Carga un archivo .mat de un gesto y retorna todas las repeticiones EMG estandarizadas.
    """
    try:
        # simplify_cells=True elimina la necesidad de funciones complejas para desanidar
        raw = scipy.io.loadmat(str(mat_path), simplify_cells=True)
    except Exception as e:
        print(f"[mat_loader] ERROR al leer {mat_path}: {e}")
        return None

    try:
        # Acceso directo gracias a simplify_cells=True
        gesture_data = raw['reps'][gesture_name]['data']

        # Si solo hay una repetición, scipy lo lee como un dict, no como una lista
        if isinstance(gesture_data, dict):
            gesture_data = [gesture_data]

        n_reps = len(gesture_data)
        emg_list = []

        for i in range(n_reps):
            item = gesture_data[i]
            
            # Skip empty elements
            if isinstance(item, np.ndarray) and item.size == 0:
                continue
            
            if hasattr(item, 'emg'):
                emg_raw = item.emg
            elif isinstance(item, dict) and 'emg' in item:
                emg_raw = item['emg']
            else:
                try:
                    emg_raw = item['emg']
                except (IndexError, TypeError, KeyError, ValueError):
                    emg_raw = item

            try:
                emg = np.array(emg_raw, dtype=np.float32)
                if emg.size > 0 and len(emg.shape) >= 2:
                    emg_list.append(emg)
            except Exception as e:
                pass

        if not emg_list:
            print(f"[mat_loader] WARN: 0 repeticiones encontradas en {mat_path}")
            return None

        # CORRECCIÓN DE SHAPE: Truncar todas las repeticiones a la longitud mínima 
        # para evitar el ValueError en np.stack por diferencias de muestreo.
        min_len = min(e.shape[0] for e in emg_list)
        emg_list = [e[:min_len, :] for e in emg_list]

        return np.stack(emg_list, axis=0)                  # (N_reps, min_len, 8)

    except Exception as e:
        import traceback
        print(f"[mat_loader] ERROR al parsear {mat_path} (gesto '{gesture_name}'):")
        traceback.print_exc()
        return None

def load_user_session(user_dir: Path,
                      gestures: List[str] = GESTURES
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Carga todas las repeticiones EMG de un usuario para una sesión (un mes).
    """
    all_emg = []
    all_labels = []

    for gesture in gestures:
        mat_file = user_dir / f"{gesture}.mat"
        if not mat_file.exists():
            warnings.warn(f"Archivo no encontrado: {mat_file}")
            continue

        emg_arr = _load_gesture_mat(mat_file, gesture)  # (N_reps, T, C)
        if emg_arr is None or len(emg_arr) == 0:
            continue

        all_emg.append(emg_arr)
        all_labels.extend([gesture] * emg_arr.shape[0])

    if not all_emg:
        return np.array([]), np.array([])

    # --- NUEVA CORRECCIÓN ---
    # Encontrar la longitud mínima de tiempo (dimensión 1) entre TODOS los gestos
    min_time_len = min(arr.shape[1] for arr in all_emg)
    
    # Recortar la dimensión de tiempo de todos los gestos a esa longitud mínima
    all_emg = [arr[:, :min_time_len, :] for arr in all_emg]
    # ------------------------

    X_raw = np.concatenate(all_emg, axis=0)   # (N_total, min_time_len, C)
    y = np.array(all_labels)
    return X_raw, y

def load_month(month_dir: Path,
               gestures: List[str] = GESTURES
               ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Carga todos los usuarios de un mes.
    """
    result = {}
    for user_dir in sorted(month_dir.iterdir()):
        if not user_dir.is_dir():
            continue
        user_id = user_dir.name   
        X_raw, y = load_user_session(user_dir, gestures=gestures)
        if len(X_raw) > 0:
            result[user_id] = (X_raw, y)
            print(f"  Cargado {user_id}: {X_raw.shape[0]} repeticiones, "
                  f"{X_raw.shape[1]} muestras × {X_raw.shape[2]} canales")
        else:
            warnings.warn(f"Sin datos para {user_id} en {month_dir.name}")
    return result

def load_all_sessions(base_dir: Path,
                      gestures: List[str] = GESTURES,
                      month_prefix: str = 'Mes'
                      ) -> Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """
    Carga todos los meses disponibles en la carpeta base.
    """
    sessions = {}
    month_dirs = sorted(
        [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith(month_prefix)],
        key=lambda d: int(d.name.replace(month_prefix, ''))
    )

    for month_dir in month_dirs:
        month_name = month_dir.name
        print(f"\n📅 Cargando {month_name}...")
        sessions[month_name] = load_month(month_dir, gestures=gestures)

    return sessions