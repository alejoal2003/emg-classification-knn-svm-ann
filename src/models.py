import os
import joblib
from typing import Dict, Any, Tuple

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# TensorFlow/Keras para ANN
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
# (Opcional, usando SciKeras o implementando GridSearchCV custom para Keras)

from src.config import Config

# ==========================================
# k-Nearest Neighbors (kNN)
# ==========================================
def get_knn_model() -> KNeighborsClassifier:
    """Retorna un estimador base kNN."""
    return KNeighborsClassifier()

def get_knn_grid() -> Dict[str, list]:
    """Retorna el param_grid para kNN definido en config."""
    return Config.KNN_PARAM_GRID

# ==========================================
# Support Vector Machine (SVM)
# ==========================================
def get_svm_model() -> SVC:
    """Retorna un estimador base SVM."""
    return SVC(probability=False, random_state=Config.RANDOM_STATE)

def get_svm_grid() -> Dict[str, list]:
    """Retorna el param_grid para SVM definido en config."""
    return Config.SVM_PARAM_GRID

# ==========================================
# Búsqueda en Grilla (Scikit-Learn)
# ==========================================
def run_grid_search(estimator, X, y, param_grid: dict, cv: int = 5) -> Tuple[Any, Dict[str, Any]]:
    """
    Ejecuta GridSearchCV sobre un estimador de scikit-learn.
    Devuelve el mejor modelo y los mejores parámetros.
    """
    grid = GridSearchCV(estimator, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1)
    grid.fit(X, y)
    print(f"Mejor score: {grid.best_score_}")
    print(f"Mejores hiperparámetros: {grid.best_params_}")
    
    return grid.best_estimator_, grid.best_params_

# ==========================================
# Artificial Neural Network (ANN)
# ==========================================
def create_ann_model(input_dim: int, n_classes: int, 
                     hidden_layers: tuple = (128, 64), 
                     activation: str = 'relu',
                     learning_rate: float = 0.001) -> Sequential:
    """
    Crea una red neuronal artificial Feed-Forward (MLP).
    Args:
        input_dim: Número de características (features).
        n_classes: Número de clases (gestos).
        hidden_layers: Tupla con neuronas por capa (ej. (128, 64)).
    """
    model = Sequential()
    
    # Capa de entrada y primera capa oculta
    model.add(Dense(hidden_layers[0], input_dim=input_dim, activation=activation))
    model.add(Dropout(0.2))
    
    # Capas ocultas adicionales
    for units in hidden_layers[1:]:
        model.add(Dense(units, activation=activation))
        model.add(Dropout(0.2))
        
    # Capa de salida (Softmax para clasificación multiclase)
    model.add(Dense(n_classes, activation='softmax'))
    
    # Compilación
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model

# ==========================================
# Guardar/Cargar modelos
# ==========================================
def save_sklearn_model(model, filepath: str):
    """Guarda un modelo de scikit-learn (kNN, SVM) con Joblib."""
    joblib.dump(model, filepath)

def load_sklearn_model(filepath: str):
    """Carga un modelo de scikit-learn."""
    return joblib.load(filepath)

def save_keras_model(model, filepath: str):
    """Guarda un modelo ANN de TensorFlow."""
    model.save(filepath)

def load_keras_model(filepath: str):
    """Carga un modelo ANN de TensorFlow."""
    return tf.keras.models.load_model(filepath)
