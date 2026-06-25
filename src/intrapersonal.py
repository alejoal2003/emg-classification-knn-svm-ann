"""
intrapersonal.py — Funciones para el análisis intrapersonal longitudinal.

Cada usuario tiene sus propios 3 modelos (kNN, SVM, ANN) entrenados
únicamente con sus datos de Mes0 y evaluados contra sus meses siguientes.
Los hiperparámetros están fijos (optimizados previamente con EMG-EPN-612).
"""

import numpy as np
import joblib
import os
import warnings
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

from src.feature_extraction import extract_features

# ============================================================================
# Constantes — Hiperparámetros óptimos (determinados en notebooks 04–06)
# ============================================================================

KNN_PARAMS = dict(n_neighbors=21, metric='minkowski', p=1, weights='distance')
SVM_PARAMS = dict(C=100, kernel='rbf', gamma=0.001, probability=True)

# Clases fijas del proyecto
CLASS_NAMES = ['fist', 'open', 'pinch', 'waveIn', 'waveOut', 'relax']

# Parámetros por defecto para ANN (configurables)
DEFAULT_ANN_PARAMS = dict(
    epochs=200,
    batch_size=32,
    validation_split=0.15,
    patience_es=20,
    patience_lr=7,
    lr_factor=0.5,
    verbose=0,
)

RANDOM_STATE = 42


# ============================================================================
# 1. Extracción de features por usuario y mes
# ============================================================================

def build_features_by_user_month(
    sessions: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]
) -> Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """
    Transforma la estructura de sesiones cargadas por load_all_sessions()
    en una estructura indexada por usuario y mes con features extraídas.

    Args:
        sessions: dict[mes][user_id] = (X_raw, y_labels)
                  X_raw shape (n_reps, T, C), y_labels shape (n_reps,)

    Returns:
        features_by_user_month: dict[user_id][mes] = (X_features, y_labels)
                                X_features shape (n_reps, 72), y shape (n_reps,)
    """
    features_by_user_month = {}

    for mes, users_data in sessions.items():
        for user_id, (X_raw, y) in users_data.items():
            if user_id not in features_by_user_month:
                features_by_user_month[user_id] = {}

            n_reps = X_raw.shape[0]
            # Extraer features para cada repetición
            X_features = np.array([
                extract_features(X_raw[i]) for i in range(n_reps)
            ])
            features_by_user_month[user_id][mes] = (X_features, y)

    # Ordenar meses dentro de cada usuario
    for user_id in features_by_user_month:
        sorted_months = dict(sorted(
            features_by_user_month[user_id].items(),
            key=lambda item: int(item[0].replace('Mes', ''))
        ))
        features_by_user_month[user_id] = sorted_months

    return features_by_user_month


# ============================================================================
# 2. Construcción del modelo ANN con la arquitectura fija
# ============================================================================

def _build_ann(input_dim: int, n_classes: int) -> Sequential:
    """
    Construye la ANN con la arquitectura óptima determinada en notebook 06:
    Dense(256) → BN → Drop(0.3) → Dense(128) → BN → Drop(0.3) →
    Dense(64) → BN → Drop(0.2) → Dense(5, softmax)
    """
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(n_classes, activation='softmax'),
    ])
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ============================================================================
# 3. Entrenamiento de modelos por usuario
# ============================================================================

def train_user_models(
    X_mes0: np.ndarray,
    y_mes0: np.ndarray,
    class_names: List[str] = None,
    ann_params: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Entrena los 3 modelos (kNN, SVM, ANN) para un usuario usando solo su Mes0.

    Args:
        X_mes0: array (n_reps, 72) — features del Mes0 del usuario
        y_mes0: array (n_reps,) — etiquetas string ('fist', 'open', ...)
        class_names: lista de clases (por defecto CLASS_NAMES)
        ann_params: dict con parámetros de entrenamiento de ANN (opcional)

    Returns:
        dict con keys 'kNN', 'SVM', 'ANN', cada uno conteniendo:
        - 'model': el modelo entrenado (Pipeline para kNN/SVM, Sequential para ANN)
        - 'scaler': StandardScaler ajustado (solo para ANN)
        - 'label_encoder': LabelEncoder ajustado (solo para ANN)
    """
    if class_names is None:
        class_names = CLASS_NAMES
    if ann_params is None:
        ann_params = DEFAULT_ANN_PARAMS.copy()

    models = {}

    # ── kNN ─────────────────────────────────────────────────────
    knn_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(**KNN_PARAMS))
    ])
    knn_pipe.fit(X_mes0, y_mes0)
    models['kNN'] = {'model': knn_pipe}

    # ── SVM ─────────────────────────────────────────────────────
    svm_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(**SVM_PARAMS))
    ])
    svm_pipe.fit(X_mes0, y_mes0)
    models['SVM'] = {'model': svm_pipe}

    # ── ANN ─────────────────────────────────────────────────────
    # Scaler y LabelEncoder propios del usuario
    ann_scaler = StandardScaler()
    X_scaled = ann_scaler.fit_transform(X_mes0)

    ann_le = LabelEncoder()
    ann_le.fit(class_names)  # Ajustar con las 5 clases fijas
    y_encoded = ann_le.transform(y_mes0)
    y_cat = to_categorical(y_encoded, num_classes=len(class_names))

    n_features = X_scaled.shape[1]
    n_classes = len(class_names)

    ann_model = _build_ann(n_features, n_classes)

    callbacks = [
        EarlyStopping(
            patience=ann_params.get('patience_es', 20),
            restore_best_weights=True,
            monitor='val_loss'
        ),
        ReduceLROnPlateau(
            patience=ann_params.get('patience_lr', 7),
            factor=ann_params.get('lr_factor', 0.5),
            monitor='val_loss'
        ),
    ]

    ann_model.fit(
        X_scaled, y_cat,
        epochs=ann_params.get('epochs', 200),
        batch_size=ann_params.get('batch_size', 32),
        validation_split=ann_params.get('validation_split', 0.15),
        callbacks=callbacks,
        verbose=ann_params.get('verbose', 0),
    )

    models['ANN'] = {
        'model': ann_model,
        'scaler': ann_scaler,
        'label_encoder': ann_le,
    }

    return models


# ============================================================================
# 4. Evaluación de un modelo sobre un mes
# ============================================================================

def evaluate_user_model(
    model_info: Dict[str, Any],
    X_mes: np.ndarray,
    y_mes: np.ndarray,
    model_type: str,
) -> Dict[str, float]:
    """
    Evalúa un modelo entrenado sobre los datos de un mes dado.

    Args:
        model_info: dict devuelto por train_user_models[model_type]
        X_mes: features del mes a evaluar (n_reps, 72)
        y_mes: etiquetas verdaderas (n_reps,) — strings
        model_type: 'kNN', 'SVM' o 'ANN'

    Returns:
        dict con 'accuracy', 'precision', 'recall', 'f1'
    """
    model = model_info['model']

    if model_type == 'ANN':
        scaler = model_info['scaler']
        le = model_info['label_encoder']
        X_scaled = scaler.transform(X_mes)
        y_probs = model.predict(X_scaled, verbose=0)
        y_pred_enc = np.argmax(y_probs, axis=1)
        y_pred = le.inverse_transform(y_pred_enc)
    else:
        # kNN y SVM usan Pipeline con StandardScaler interno
        y_pred = model.predict(X_mes)

    acc = accuracy_score(y_mes, y_pred)
    prec = precision_score(y_mes, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_mes, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_mes, y_pred, average='macro', zero_division=0)

    return {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
    }


# ============================================================================
# 5. Validación cruzada estratificada para Mes0
# ============================================================================

def cross_validate_user_mes0(
    X_mes0: np.ndarray,
    y_mes0: np.ndarray,
    class_names: List[str] = None,
    n_splits: int = 5,
    ann_params: Dict[str, Any] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Evalúa los 3 modelos sobre Mes0 mediante validación cruzada estratificada.
    Así el accuracy reportado para Mes0 es honesto (no sobre el training set).

    Si alguna clase tiene menos muestras que n_splits, reduce el número de
    folds automáticamente y lo informa por consola.

    Args:
        X_mes0: features del Mes0 (n_reps, 72)
        y_mes0: etiquetas string (n_reps,)
        class_names: lista de clases (default CLASS_NAMES)
        n_splits: número de folds (default 5)
        ann_params: parámetros configurables de ANN

    Returns:
        dict[model_name] = {
            'accuracy': float (media),
            'std': float,
            'scores': np.ndarray (por fold),
            'n_splits_used': int
        }
    """
    if class_names is None:
        class_names = CLASS_NAMES
    if ann_params is None:
        ann_params = DEFAULT_ANN_PARAMS.copy()

    # Verificar que hay suficientes muestras por clase para el número de folds
    unique, counts = np.unique(y_mes0, return_counts=True)
    min_samples_per_class = counts.min()
    actual_splits = n_splits

    if min_samples_per_class < n_splits:
        actual_splits = max(2, min_samples_per_class)
        print(f"  ⚠️  Clase con menos muestras ({min_samples_per_class}) < {n_splits} folds. "
              f"Reduciendo a {actual_splits} folds.")

    cv = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=RANDOM_STATE)

    results = {}

    # ── kNN CV ──────────────────────────────────────────────────
    knn_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(**KNN_PARAMS))
    ])
    knn_scores = cross_val_score(knn_pipe, X_mes0, y_mes0, cv=cv, scoring='accuracy')
    results['kNN'] = {
        'accuracy': float(knn_scores.mean()),
        'std': float(knn_scores.std()),
        'scores': knn_scores,
        'n_splits_used': actual_splits,
    }

    # ── SVM CV ──────────────────────────────────────────────────
    svm_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(**SVM_PARAMS))
    ])
    svm_scores = cross_val_score(svm_pipe, X_mes0, y_mes0, cv=cv, scoring='accuracy')
    results['SVM'] = {
        'accuracy': float(svm_scores.mean()),
        'std': float(svm_scores.std()),
        'scores': svm_scores,
        'n_splits_used': actual_splits,
    }

    # ── ANN CV (manual, porque Keras no se integra en cross_val_score) ──
    le = LabelEncoder()
    le.fit(class_names)
    y_encoded = le.transform(y_mes0)

    ann_scores = []
    for fold_i, (train_idx, val_idx) in enumerate(cv.split(X_mes0, y_mes0)):
        X_tr, X_vl = X_mes0[train_idx], X_mes0[val_idx]
        y_tr, y_vl = y_encoded[train_idx], y_encoded[val_idx]

        sc_fold = StandardScaler()
        X_tr_sc = sc_fold.fit_transform(X_tr)
        X_vl_sc = sc_fold.transform(X_vl)

        n_classes = len(class_names)
        ann_fold = _build_ann(X_tr_sc.shape[1], n_classes)

        callbacks = [
            EarlyStopping(
                patience=ann_params.get('patience_es', 20),
                restore_best_weights=True,
                monitor='val_loss'
            ),
            ReduceLROnPlateau(
                patience=ann_params.get('patience_lr', 7),
                factor=ann_params.get('lr_factor', 0.5),
                monitor='val_loss'
            ),
        ]

        ann_fold.fit(
            X_tr_sc, to_categorical(y_tr, n_classes),
            epochs=ann_params.get('epochs', 200),
            batch_size=ann_params.get('batch_size', 32),
            validation_split=ann_params.get('validation_split', 0.15),
            callbacks=callbacks,
            verbose=0,
        )

        y_pred_fold = np.argmax(ann_fold.predict(X_vl_sc, verbose=0), axis=1)
        fold_acc = accuracy_score(y_vl, y_pred_fold)
        ann_scores.append(fold_acc)

    ann_scores = np.array(ann_scores)
    results['ANN'] = {
        'accuracy': float(ann_scores.mean()),
        'std': float(ann_scores.std()),
        'scores': ann_scores,
        'n_splits_used': actual_splits,
    }

    return results


# ============================================================================
# 6. Persistencia de modelos
# ============================================================================

def save_user_models(
    models_dict: Dict[str, Any],
    user_id: str,
    base_dir: Path,
) -> Dict[str, str]:
    """
    Guarda los 3 modelos entrenados de un usuario.

    Estructura de salida:
        base_dir/knn/{user_id}.pkl
        base_dir/svm/{user_id}.pkl
        base_dir/ann/{user_id}.keras
        base_dir/ann/{user_id}_scaler.pkl
        base_dir/ann/{user_id}_encoder.pkl

    Returns:
        dict con las rutas de los archivos guardados
    """
    saved_paths = {}

    # ── kNN ──
    knn_dir = base_dir / 'knn'
    knn_dir.mkdir(parents=True, exist_ok=True)
    knn_path = knn_dir / f'{user_id}.pkl'
    joblib.dump(models_dict['kNN']['model'], knn_path)
    saved_paths['kNN'] = str(knn_path)

    # ── SVM ──
    svm_dir = base_dir / 'svm'
    svm_dir.mkdir(parents=True, exist_ok=True)
    svm_path = svm_dir / f'{user_id}.pkl'
    joblib.dump(models_dict['SVM']['model'], svm_path)
    saved_paths['SVM'] = str(svm_path)

    # ── ANN ──
    ann_dir = base_dir / 'ann'
    ann_dir.mkdir(parents=True, exist_ok=True)
    ann_model_path = ann_dir / f'{user_id}.keras'
    ann_scaler_path = ann_dir / f'{user_id}_scaler.pkl'
    ann_encoder_path = ann_dir / f'{user_id}_encoder.pkl'

    models_dict['ANN']['model'].save(ann_model_path)
    joblib.dump(models_dict['ANN']['scaler'], ann_scaler_path)
    joblib.dump(models_dict['ANN']['label_encoder'], ann_encoder_path)
    saved_paths['ANN'] = str(ann_model_path)
    saved_paths['ANN_scaler'] = str(ann_scaler_path)
    saved_paths['ANN_encoder'] = str(ann_encoder_path)

    return saved_paths


def load_user_models(
    user_id: str,
    base_dir: Path,
    class_names: List[str] = None,
) -> Dict[str, Any]:
    """
    Carga los 3 modelos entrenados de un usuario desde disco.

    Returns:
        dict con la misma estructura que train_user_models().
    """
    if class_names is None:
        class_names = CLASS_NAMES

    models = {}

    # ── kNN ──
    knn_path = base_dir / 'knn' / f'{user_id}.pkl'
    models['kNN'] = {'model': joblib.load(knn_path)}

    # ── SVM ──
    svm_path = base_dir / 'svm' / f'{user_id}.pkl'
    models['SVM'] = {'model': joblib.load(svm_path)}

    # ── ANN ──
    ann_model_path = base_dir / 'ann' / f'{user_id}.keras'
    ann_scaler_path = base_dir / 'ann' / f'{user_id}_scaler.pkl'
    ann_encoder_path = base_dir / 'ann' / f'{user_id}_encoder.pkl'

    models['ANN'] = {
        'model': tf.keras.models.load_model(ann_model_path),
        'scaler': joblib.load(ann_scaler_path),
        'label_encoder': joblib.load(ann_encoder_path),
    }

    return models
