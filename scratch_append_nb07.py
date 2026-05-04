import nbformat as nbf

nb_path = r"c:\Users\aseba\TIC\emg-classification-knn-svm-ann\notebooks\07_evaluacion_longitudinal.ipynb"

# Leer el notebook
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

# Crear celdas
md_cell = nbf.v4.new_markdown_cell("""## 6. Demostración de Reconocimiento (Onset/Offset)
Tal como lo requiere el plan de tesis, un sistema completo HGR no solo debe clasificar qué gesto se está realizando, sino **reconocer** cuándo empieza y termina la actividad muscular. 
El siguiente bloque utiliza la envolvente RMS y un umbral dinámico de ruido para detectar los puntos de **Onset** (inicio) y **Offset** (fin) de un gesto de prueba.""")

code = """# ==============================================================================
# DEMOSTRACIÓN DE RECONOCIMIENTO DE ACTIVIDAD (ONSET/OFFSET)
# ==============================================================================
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Asegurar que el modulo src se pueda importar
sys.path.append(os.path.abspath('..'))
from src.activity_detection import plot_onset_offset

# Cargar un archivo de ejemplo (ej. user12, Mes0, gesto fist)
base_path = "../data/longitudinal/usuarios/Mes0/user12/fist.mat"
if not os.path.exists(base_path):
    base_path = "data/longitudinal/usuarios/Mes0/user12/fist.mat"

try:
    raw = scipy.io.loadmat(base_path, simplify_cells=True)
    gesture_data = raw['reps']['fist']['data']
    
    # Manejar variabilidad en la estructura del archivo .mat
    if isinstance(gesture_data, dict):
        gesture_data = [gesture_data]
    elif isinstance(gesture_data, np.ndarray) and len(gesture_data) > 0 and isinstance(gesture_data[0], dict) and 'emg' not in gesture_data[0]:
        pass # Por si la estructura es diferente

    emg_raw = gesture_data[0]['emg'] if 'emg' in gesture_data[0] else gesture_data[0]
    signal = np.array(emg_raw, dtype=np.float32)
    
    # Generar gráfica de Onset/Offset
    print("Analizando la primera repetición del gesto Fist (Usuario 12 - Mes 0)...")
    segments = plot_onset_offset(
        signal=signal,
        gesture_name="Fist (Puño) - Demostración",
        window_size=40,
        noise_samples=100,  # Estimamos ruido base en el primer 0.5 seg
        multiplier=3.0,
        sample_rate=200,
        save_path=None
    )
except Exception as e:
    print(f"Error al cargar o procesar la señal para la demostración: {e}")"""
code_cell = nbf.v4.new_code_cell(code)

# Añadir las celdas al notebook
nb.cells.extend([md_cell, code_cell])

# Guardar el notebook
with open(nb_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Celdas añadidas correctamente al Notebook 07.")
