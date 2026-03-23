import os
from pathlib import Path

class Config:
    """Configuración central para el proyecto de clasificación EMG."""
    
    # Rutas base
    # Si estás en Google Colab, cambia BASE_DIR a la ruta de tu proyecto en Drive
    # Ejemplo: BASE_DIR = Path('/content/drive/MyDrive/Tesis_HGR/emg-classification-knn-svm-ann')
    BASE_DIR = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    DATA_DIR = BASE_DIR / 'data'
    TRAIN_JSON_DIR = DATA_DIR / 'trainingJSON'
    TEST_JSON_DIR = DATA_DIR / 'testingJSON'
    PROCESSED_DIR = DATA_DIR / 'processed'
    
    # Parámetros de señal de EMG-EPN-612
    SAMPLE_RATE = 200  # Hz
    N_CHANNELS = 8
    
    # Clases (Gestos)
    GESTURES = ['fist', 'open', 'pinch', 'waveIn', 'waveOut']
    INCLUDE_NOGESTURE = False  # Cambiar a True si se quiere incluir la clase de reposo
    
    # Parámetros de ventaneo (inspirado en la guía de MATLAB)
    WINDOW_SIZE = 300  # 1.5 segundos a 200 Hz
    WINDOW_STEP = 30   # 150 ms (Overlap de 270 muestras)
    
    # Filtro
    LOW_CUTOFF = 20    # Hz
    HIGH_CUTOFF = 90   # Hz
    FILTER_ORDER = 5
    
    # Grid Search Hiperparámetros
    KNN_PARAM_GRID = {
        'n_neighbors': [3, 5, 7, 9, 11, 15],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    
    SVM_PARAM_GRID = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['rbf', 'linear', 'poly'],
        'gamma': ['scale', 'auto', 0.01, 0.001]
    }
    
    # Semilla para reproducibilidad
    RANDOM_STATE = 42

# Asegurar que las carpetas de salida existan
os.makedirs(Config.PROCESSED_DIR, exist_ok=True)
