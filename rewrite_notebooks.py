import json, os

REPO = r"c:\Users\aseba\TIC\emg-classification-knn-svm-ann\notebooks"

# ==============================================================
# BLOQUE BASE DE SETUP (compartido por todos los notebooks)
# ==============================================================
def colab_setup_cell():
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os, sys, numpy as np, pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "\n",
            "try:\n",
            "    from google.colab import drive\n",
            "    drive.mount('/content/drive', force_remount=True)\n",
            "    import glob\n",
            "    paths = glob.glob('/content/drive/**/emg-classification-knn-svm-ann', recursive=True)\n",
            "    pc = [p for p in paths if 'Othercomputers' in p or 'Ordenadores' in p]\n",
            "    PROJECT_PATH = sorted(pc or paths, key=len)[0] if paths else '/content/drive/Othercomputers/My PC/emg-classification-knn-svm-ann'\n",
            "    os.chdir(PROJECT_PATH)\n",
            "    sys.path.insert(0, PROJECT_PATH)\n",
            "except:\n",
            "    if 'notebooks' in os.getcwd(): os.chdir('..')\n",
            "    sys.path.insert(0, os.getcwd())\n",
            "\n",
            "print('CWD:', os.getcwd())\n",
        ]
    }

# ==============================================================
# NOTEBOOK 02 — EDA
# ==============================================================
nb02_cells = [
    {"cell_type":"markdown","metadata":{},"source":[
        "# Notebook 2 — Exploración de Datos (EDA)\n\n",
        "Visualiza señales EMG crudas, segmentadas y filtradas."
    ]},
    colab_setup_cell(),
    {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[
        "from src.config import Config\n",
        "from src.preprocessing import preprocess_pipeline, bandpass_filter, segment_gesture\n",
        "from src.visualization import plot_signal\n",
        "\n",
        "# Cargar DataFrame del Notebook 1\n",
        "db_path = Config.PROCESSED_DIR / 'notebook1' / 'samples_full.pkl'\n",
        "df = pd.read_pickle(db_path)\n",
        "print(f'Dataset cargado: {len(df)} muestras.')\n",
        "df.head(3)\n",
    ]},
    {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[
        "# Resumen de muestras por gesto y split\n",
        "resumen = df.groupby(['gesture_name','split']).size().unstack(fill_value=0)\n",
        "display(resumen)\n",
    ]},
    {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[
        "# Seleccionar una muestra al azar y visualizarla\n",
        "sample = df.sample(1, random_state=42).iloc[0]\n",
        "signal_raw   = sample['signal_raw']        # shape [T, 8]\n",
        "gt_idx       = sample['ground_truth_index'] # [inicio, fin]\n",
        "gesture      = sample['gesture_name']\n",
        "\n",
        "print(f\"Gesto: {gesture}  |  Señal completa: {signal_raw.shape}\")\n",
        "fig = plot_signal(signal_raw, title=f'Señal EMG Cruda — {gesture}', fs=Config.SAMPLE_RATE)\n",
        "plt.show()\n",
    ]},
    {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[
        "# Señal recortada (solo la acción) y filtrada\n",
        "signal_seg = segment_gesture(signal_raw, gt_idx)\n",
        "signal_filt = bandpass_filter(signal_seg)\n",
        "print(f'Señal recortada: {signal_seg.shape}  |  Filtrada: {signal_filt.shape}')\n",
        "fig2 = plot_signal(signal_filt, title=f'Señal Recortada + Filtro Pasabanda — {gesture}', fs=Config.SAMPLE_RATE)\n",
        "plt.show()\n",
    ]},
]

# ==============================================================
# NOTEBOOK 03 — Feature Extraction
# ==============================================================
nb03_cells = [
    {"cell_type":"markdown","metadata":{},"source":[
        "# Notebook 3 — Extracción de Características (Features)\n\n",
        "Genera la matriz X y el vector y que consumirán los modelos ML."
    ]},
    colab_setup_cell(),
    {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[
        "from tqdm.auto import tqdm\n",
        "from src.config import Config\n",
        "from src.preprocessing import preprocess_pipeline, segment_gesture\n",
        "from src.feature_extraction import extract_features\n",
        "\n",
        "db_path = Config.PROCESSED_DIR / 'notebook1' / 'samples_full.pkl'\n",
        "df = pd.read_pickle(db_path)\n",
        "print(f'Muestras totales: {len(df)}')\n",
    ]},
    {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[
        "# Preprocesamiento y extracción de features por muestra\n",
        "feature_rows = []\n",
        "labels = []\n",
        "users = []\n",
        "splits = []\n",
        "\n",
        "for _, row in tqdm(df.iterrows(), total=len(df), desc='Extrayendo features'):\n",
        "    sig   = segment_gesture(row['signal_raw'], row['ground_truth_index'])\n",
        "    sig_p = preprocess_pipeline(sig)\n",
        "    feat  = extract_features(sig_p)  # shape (48,)\n",
        "    feature_rows.append(feat)\n",
        "    labels.append(row['gesture_name'])\n",
        "    users.append(row['user_id'])\n",
        "    splits.append(row['split'])\n",
        "\n",
        "X = np.array(feature_rows)\n",
        "y = np.array(labels)\n",
        "print(f'X shape: {X.shape}  |  y shape: {y.shape}')\n",
    ]},
    {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[
        "# Guardar\n",
        "out_dir = Config.PROCESSED_DIR / 'features'\n",
        "os.makedirs(out_dir, exist_ok=True)\n",
        "np.savez_compressed(out_dir / 'emg_features_base.npz', X=X, y=y)\n",
        "np.savez_compressed(out_dir / 'emg_metadata_base.npz', y=y, users=np.array(users), splits=np.array(splits))\n",
        "print(f'Guardado en: {out_dir}')\n",
    ]},
]

# ==============================================================
# HELPER — celdas de carga de features (base común 04-06)
# ==============================================================
def load_features_cells():
    return [
        {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[
            "from src.config import Config\n",
            "from src.evaluation import get_confusion_matrix, compute_metrics\n",
            "from src.visualization import plot_confusion_matrix\n",
            "from sklearn.metrics import classification_report\n",
            "\n",
            "features_dir = Config.PROCESSED_DIR / 'features'\n",
            "data = np.load(features_dir / 'emg_features_base.npz', allow_pickle=True)\n",
            "meta = np.load(features_dir / 'emg_metadata_base.npz', allow_pickle=True)\n",
            "\n",
            "X      = data['X']\n",
            "y      = data['y']\n",
            "splits = meta['splits']\n",
            "\n",
            "train_idx = np.where(splits == 'train_samples')[0]\n",
            "test_idx  = np.where(splits == 'test_samples')[0]\n",
            "\n",
            "X_train, y_train = X[train_idx], y[train_idx]\n",
            "X_test,  y_test  = X[test_idx],  y[test_idx]\n",
            "\n",
            "class_names = sorted(np.unique(y).tolist())\n",
            "print(f'Train: {X_train.shape}  |  Test: {X_test.shape}')\n",
            "print('Clases:', class_names)\n",
        ]}
    ]

# ==============================================================
# NOTEBOOK 04 — kNN
# ==============================================================
nb04_cells = [
    {"cell_type":"markdown","metadata":{},"source":["# Notebook 4 — Modelo k-Nearest Neighbors (kNN)"]},
    colab_setup_cell(),
] + load_features_cells() + [
    {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[
        "import joblib\n",
        "from sklearn.neighbors import KNeighborsClassifier\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "from sklearn.pipeline import Pipeline\n",
        "from sklearn.model_selection import GridSearchCV\n",
        "\n",
        "pipeline_knn = Pipeline([\n",
        "    ('scaler', StandardScaler()),\n",
        "    ('knn',    KNeighborsClassifier())\n",
        "])\n",
        "\n",
        "param_grid = {\n",
        "    'knn__n_neighbors': [3, 5, 7, 9],\n",
        "    'knn__weights':     ['uniform', 'distance'],\n",
        "    'knn__metric':      ['euclidean', 'manhattan'],\n",
        "}\n",
        "\n",
        "print('Iniciando Grid Search...')\n",
        "grid_knn = GridSearchCV(pipeline_knn, param_grid, cv=5, n_jobs=-1, verbose=1)\n",
        "grid_knn.fit(X_train, y_train)\n",
        "print('Mejores parámetros:', grid_knn.best_params_)\n",
    ]},
    {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[
        "best_knn = grid_knn.best_estimator_\n",
        "y_pred   = best_knn.predict(X_test)\n",
        "\n",
        "print('Reporte kNN:\\n')\n",
        "print(classification_report(y_test, y_pred, target_names=class_names))\n",
        "\n",
        "cm = get_confusion_matrix(y_test, y_pred)\n",
        "plot_confusion_matrix(cm, class_names, title='Matriz de Confusión — kNN')\n",
    ]},
    {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[
        "models_dir = Config.PROCESSED_DIR / 'models'\n",
        "os.makedirs(models_dir, exist_ok=True)\n",
        "joblib.dump(best_knn, models_dir / 'best_knn_model.pkl')\n",
        "print('Modelo kNN guardado.')\n",
    ]},
]

# ==============================================================
# NOTEBOOK 05 — SVM
# ==============================================================
nb05_cells = [
    {"cell_type":"markdown","metadata":{},"source":["# Notebook 5 — Modelo Support Vector Machine (SVM)"]},
    colab_setup_cell(),
] + load_features_cells() + [
    {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[
        "import joblib\n",
        "from sklearn.svm import SVC\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "from sklearn.pipeline import Pipeline\n",
        "from sklearn.model_selection import GridSearchCV\n",
        "\n",
        "pipeline_svm = Pipeline([\n",
        "    ('scaler', StandardScaler()),\n",
        "    ('svm',    SVC(probability=True, random_state=42))\n",
        "])\n",
        "\n",
        "param_grid = {\n",
        "    'svm__C':      [0.1, 1, 10],\n",
        "    'svm__kernel': ['rbf', 'linear'],\n",
        "    'svm__gamma':  ['scale', 'auto'],\n",
        "}\n",
        "\n",
        "print('Iniciando Grid Search SVM (puede tardar varios minutos)...')\n",
        "grid_svm = GridSearchCV(pipeline_svm, param_grid, cv=3, n_jobs=-1, verbose=1)\n",
        "grid_svm.fit(X_train, y_train)\n",
        "print('Mejores parámetros:', grid_svm.best_params_)\n",
    ]},
    {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[
        "best_svm = grid_svm.best_estimator_\n",
        "y_pred   = best_svm.predict(X_test)\n",
        "\n",
        "print('Reporte SVM:\\n')\n",
        "print(classification_report(y_test, y_pred, target_names=class_names))\n",
        "\n",
        "cm = get_confusion_matrix(y_test, y_pred)\n",
        "plot_confusion_matrix(cm, class_names, title='Matriz de Confusión — SVM')\n",
    ]},
    {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[
        "models_dir = Config.PROCESSED_DIR / 'models'\n",
        "os.makedirs(models_dir, exist_ok=True)\n",
        "joblib.dump(best_svm, models_dir / 'best_svm_model.pkl')\n",
        "print('Modelo SVM guardado.')\n",
    ]},
]

# ==============================================================
# NOTEBOOK 06 — ANN
# ==============================================================
nb06_cells = [
    {"cell_type":"markdown","metadata":{},"source":["# Notebook 6 — Red Neuronal Artificial (ANN)"]},
    colab_setup_cell(),
] + load_features_cells() + [
    {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[
        "import joblib\n",
        "import tensorflow as tf\n",
        "from tensorflow.keras.models import Sequential\n",
        "from tensorflow.keras.layers import Dense, Dropout, Input\n",
        "from tensorflow.keras.callbacks import EarlyStopping\n",
        "from sklearn.preprocessing import StandardScaler, LabelEncoder\n",
        "\n",
        "print(f'TensorFlow: {tf.__version__}')\n",
        "\n",
        "scaler  = StandardScaler()\n",
        "encoder = LabelEncoder()\n",
        "\n",
        "X_train_s = scaler.fit_transform(X_train)\n",
        "X_test_s  = scaler.transform(X_test)\n",
        "y_train_e = encoder.fit_transform(y_train)\n",
        "y_test_e  = encoder.transform(y_test)\n",
        "\n",
        "n_classes = len(class_names)\n",
        "n_feats   = X_train_s.shape[1]\n",
        "print(f'Input dim: {n_feats}  |  Classes: {n_classes}')\n",
    ]},
    {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[
        "model = Sequential([\n",
        "    Input(shape=(n_feats,)),\n",
        "    Dense(128, activation='relu'),\n",
        "    Dropout(0.3),\n",
        "    Dense(64,  activation='relu'),\n",
        "    Dropout(0.3),\n",
        "    Dense(n_classes, activation='softmax'),\n",
        "])\n",
        "model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])\n",
        "model.summary()\n",
    ]},
    {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[
        "early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)\n",
        "\n",
        "history = model.fit(\n",
        "    X_train_s, y_train_e,\n",
        "    epochs=100, batch_size=32,\n",
        "    validation_split=0.1,\n",
        "    callbacks=[early_stop],\n",
        "    verbose=1\n",
        ")\n",
    ]},
    {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[
        "# Curvas de aprendizaje\n",
        "from src.visualization import plot_learning_curve_keras\n",
        "plot_learning_curve_keras(history)\n",
    ]},
    {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[
        "y_pred_enc = np.argmax(model.predict(X_test_s), axis=1)\n",
        "y_pred     = encoder.inverse_transform(y_pred_enc)\n",
        "\n",
        "print('Reporte ANN:\\n')\n",
        "print(classification_report(y_test, y_pred, target_names=class_names))\n",
        "\n",
        "cm = get_confusion_matrix(y_test, y_pred)\n",
        "plot_confusion_matrix(cm, class_names, title='Matriz de Confusión — ANN')\n",
    ]},
    {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[
        "models_dir = Config.PROCESSED_DIR / 'models'\n",
        "os.makedirs(models_dir, exist_ok=True)\n",
        "model.save(models_dir / 'best_ann_model.h5')\n",
        "joblib.dump(scaler,  models_dir / 'ann_scaler.pkl')\n",
        "joblib.dump(encoder, models_dir / 'ann_encoder.pkl')\n",
        "print(f'Modelos guardados en {models_dir}')\n",
    ]},
]

# ==============================================================
# GUARDAR TODOS
# ==============================================================
notebooks = {
    "02_exploracion_datos.ipynb":  nb02_cells,
    "03_extraccion_features.ipynb": nb03_cells,
    "04_modelo_knn.ipynb":         nb04_cells,
    "05_modelo_svm.ipynb":         nb05_cells,
    "06_modelo_ann.ipynb":         nb06_cells,
}

for fname, cells in notebooks.items():
    nb = {"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
    with open(os.path.join(REPO, fname), "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"  ✓ {fname}")

print("\nTodos los notebooks reescritos.")
