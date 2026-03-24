import json
import os

nb_base_setup = [
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "import sys\n",
            "import pandas as pd\n",
            "import numpy as np\n",
            "import joblib\n",
            "from sklearn.model_selection import GridSearchCV, PredefinedSplit\n",
            "from sklearn.metrics import classification_report\n",
            "import matplotlib.pyplot as plt\n",
            "\n",
            "try:\n",
            "    from google.colab import drive\n",
            "    drive.mount('/content/drive', force_remount=True)\n",
            "    import glob\n",
            "    possible_paths = glob.glob('/content/drive/**/emg-classification-knn-svm-ann', recursive=True)\n",
            "    if len(possible_paths) > 0:\n",
            "        pc_paths = [p for p in possible_paths if 'Othercomputers' in p or 'Ordenadores' in p]\n",
            "        if pc_paths: PROJECT_PATH = sorted(pc_paths, key=len)[0]\n",
            "        else: PROJECT_PATH = sorted(possible_paths, key=len)[0]\n",
            "    else:\n",
            "        PROJECT_PATH = '/content/drive/Othercomputers/My PC/emg-classification-knn-svm-ann'\n",
            "    os.chdir(PROJECT_PATH)\n",
            "    if PROJECT_PATH not in sys.path: sys.path.insert(0, PROJECT_PATH)\n",
            "except:\n",
            "    PROJECT_PATH = os.getcwd()\n",
            "    if 'notebooks' in PROJECT_PATH: os.chdir('..')\n",
            "    if os.getcwd() not in sys.path: sys.path.insert(0, os.getcwd())\n",
            "print('CWD:', os.getcwd())"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from src.config import Config\n",
            "from src.evaluation import confusion_matrix_report\n",
            "from src.visualization import plot_confusion_matrix\n",
            "\n",
            "# Cargar Features y Etiquetas generados en Notebook 03\n",
            "features_dir = Config.PROCESSED_DIR / 'features'\n",
            "data = np.load(features_dir / 'emg_features_base.npz')\n",
            "meta = np.load(features_dir / 'emg_metadata_base.npz')\n",
            "\n",
            "X = data['X']\n",
            "y = data['y']\n",
            "splits = meta['splits']\n",
            "\n",
            "print(f\"Dataset total -> X: {X.shape}, y: {y.shape}\")\n",
            "\n",
            "# Separar en Training y Testing según el split original del dataset JSON\n",
            "train_idx = np.where(splits == 'train_samples')[0]\n",
            "test_idx = np.where(splits == 'test_samples')[0]\n",
            "\n",
            "X_train, y_train = X[train_idx], y[train_idx]\n",
            "X_test, y_test = X[test_idx], y[test_idx]\n",
            "\n",
            "print(f\"Training set -> X_train: {X_train.shape}, y_train: {y_train.shape}\")\n",
            "print(f\"Testing set  -> X_test: {X_test.shape}, y_test: {y_test.shape}\")\n",
            "\n",
            "class_names = sorted(np.unique(y))\n",
            "print(\"Clases detectadas:\", class_names)"
        ]
    }
]

# 04 KNN
nb4 = nb_base_setup.copy()
nb4.insert(0, {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# Notebook 4 — Modelo k-Nearest Neighbors (kNN)\n",
        "\n",
        "En este notebook entrenamos el modelo kNN utilizando Grid Search para encontrar los hiperparámetros óptimos."
    ]
})
nb4.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "from sklearn.neighbors import KNeighborsClassifier\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "from sklearn.pipeline import Pipeline\n",
        "\n",
        "# En ML Clásico (kNN, SVM) con EMG, es crítico escalar los features numérico.\n",
        "pipeline_knn = Pipeline([\n",
        "    ('scaler', StandardScaler()),\n",
        "    ('knn', KNeighborsClassifier())\n",
        "])\n",
        "\n",
        "# Parámetros para buscar (desde config)\n",
        "param_grid = {\n",
        "    'knn__n_neighbors': [3, 5, 7, 9],\n",
        "    'knn__weights': ['uniform', 'distance'],\n",
        "    'knn__metric': ['euclidean', 'manhattan']\n",
        "}\n",
        "\n",
        "print(\"Iniciando Grid Search Cross-Validation...\")\n",
        "grid_knn = GridSearchCV(pipeline_knn, param_grid, cv=5, n_jobs=-1, verbose=2)\n",
        "grid_knn.fit(X_train, y_train)\n",
        "\n",
        "print(\"\\nMejores parámetros encontrados:\", grid_knn.best_params_)"
    ]
})
nb4.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Predicciones en el Testing set\n",
        "best_knn = grid_knn.best_estimator_\n",
        "y_pred = best_knn.predict(X_test)\n",
        "\n",
        "print(\"Reporte de Clasificación kNN:\\n\")\n",
        "print(classification_report(y_test, y_pred, target_names=class_names))\n",
        "\n",
        "cm = confusion_matrix_report(y_test, y_pred, class_names)\n",
        "plot_confusion_matrix(cm, class_names, title='Matriz de Confusión - kNN')"
    ]
})
nb4.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Guardar Modelo final\n",
        "models_dir = Config.PROCESSED_DIR / 'models'\n",
        "os.makedirs(models_dir, exist_ok=True)\n",
        "\n",
        "model_path = models_dir / 'best_knn_model.pkl'\n",
        "joblib.dump(best_knn, model_path)\n",
        "print(f\"Modelo kNN guardado en: {model_path}\")"
    ]
})

# 05 SVM
nb5 = nb_base_setup.copy()
nb5.insert(0, {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# Notebook 5 — Modelo Support Vector Machine (SVM)\n",
        "\n",
        "En este notebook entrenamos el modelo SVM utilizando Grid Search para encontrar los hiperparámetros óptimos."
    ]
})
nb5.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "from sklearn.svm import SVC\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "from sklearn.pipeline import Pipeline\n",
        "\n",
        "# Pipeline con escalador automático\n",
        "pipeline_svm = Pipeline([\n",
        "    ('scaler', StandardScaler()),\n",
        "    ('svm', SVC(probability=True, random_state=Config.RANDOM_STATE))\n",
        "])\n",
        "\n",
        "# Se reduce el param_grid temporalmente para que corra rápido en Colab para tus pruebas\n",
        "param_grid = {\n",
        "    'svm__C': [0.1, 1, 10],\n",
        "    'svm__kernel': ['rbf', 'linear'],\n",
        "    'svm__gamma': ['scale', 'auto']\n",
        "}\n",
        "\n",
        "print(\"Iniciando Grid Search Cross-Validation (esto puede tardar más que kNN)...\")\n",
        "grid_svm = GridSearchCV(pipeline_svm, param_grid, cv=3, n_jobs=-1, verbose=2)\n",
        "grid_svm.fit(X_train, y_train)\n",
        "\n",
        "print(\"\\nMejores parámetros encontrados:\", grid_svm.best_params_)"
    ]
})
nb5.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Predicciones en el Testing set\n",
        "best_svm = grid_svm.best_estimator_\n",
        "y_pred = best_svm.predict(X_test)\n",
        "\n",
        "print(\"Reporte de Clasificación SVM:\\n\")\n",
        "print(classification_report(y_test, y_pred, target_names=class_names))\n",
        "\n",
        "cm = confusion_matrix_report(y_test, y_pred, class_names)\n",
        "plot_confusion_matrix(cm, class_names, title='Matriz de Confusión - SVM')"
    ]
})
nb5.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Guardar Modelo final\n",
        "models_dir = Config.PROCESSED_DIR / 'models'\n",
        "os.makedirs(models_dir, exist_ok=True)\n",
        "\n",
        "model_path = models_dir / 'best_svm_model.pkl'\n",
        "joblib.dump(best_svm, model_path)\n",
        "print(f\"Modelo SVM guardado en: {model_path}\")"
    ]
})

# 06 ANN
nb6 = nb_base_setup.copy()
nb6.insert(0, {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# Notebook 6 — Red Neuronal Artificial (ANN)\n",
        "\n",
        "En este notebook entrenamos una red neuronal fully connected (Dense) con TensorFlow/Keras."
    ]
})
nb6.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import tensorflow as tf\n",
        "from tensorflow.keras.models import Sequential\n",
        "from tensorflow.keras.layers import Dense, Dropout, Input\n",
        "from sklearn.preprocessing import StandardScaler, LabelEncoder\n",
        "\n",
        "print(f\"TensorFlow Version: {tf.__version__}\")\n",
        "\n",
        "# 1. Escalar datos (las redes neuronales son DESTRUCTIVAMENTE sensibles a escalas)\n",
        "scaler = StandardScaler()\n",
        "X_train_scaled = scaler.fit_transform(X_train)\n",
        "X_test_scaled = scaler.transform(X_test)\n",
        "\n",
        "# 2. Codificar Etiquetas (Convertir strings a numéricas: 0, 1, 2, 3, 4)\n",
        "encoder = LabelEncoder()\n",
        "y_train_enc = encoder.fit_transform(y_train)\n",
        "y_test_enc = encoder.transform(y_test)\n",
        "\n",
        "num_classes = len(class_names)\n",
        "input_dim = X_train_scaled.shape[1]  # Ej: 48\n",
        "\n",
        "print(\"Datos escalados y etiquetas codificadas.\")"
    ]
})
nb6.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Construir Arquitectura ANN (Perceptrón Multicapa)\n",
        "def build_ann_model(input_size, num_classes):\n",
        "    model = Sequential([\n",
        "        Input(shape=(input_size,)),\n",
        "        Dense(128, activation='relu'),\n",
        "        Dropout(0.3),\n",
        "        Dense(64, activation='relu'),\n",
        "        Dropout(0.3),\n",
        "        Dense(num_classes, activation='softmax')\n",
        "    ])\n",
        "    model.compile(\n",
        "        optimizer='adam',\n",
        "        loss='sparse_categorical_crossentropy',\n",
        "        metrics=['accuracy']\n",
        "    )\n",
        "    return model\n",
        "\n",
        "ann_model = build_ann_model(input_dim, num_classes)\n",
        "ann_model.summary()"
    ]
})
nb6.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Entrenamiento de la ANN\n",
        "from tensorflow.keras.callbacks import EarlyStopping\n",
        "\n",
        "# Detener el entrenamiento temprano si dejamos de mejorar\n",
        "early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)\n",
        "\n",
        "print(\"Iniciando entrenamiento de la ANN...\")\n",
        "history = ann_model.fit(\n",
        "    X_train_scaled,\n",
        "    y_train_enc,\n",
        "    epochs=100,\n",
        "    batch_size=32,\n",
        "    validation_split=0.1,  # 10% del train como validación interna\n",
        "    callbacks=[early_stop],\n",
        "    verbose=1\n",
        ")"
    ]
})
nb6.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Gráfica de Convergencia (Pérdida y Precisión vs. Épocas)\n",
        "plt.figure(figsize=(12, 4))\n",
        "plt.subplot(1, 2, 1)\n",
        "plt.plot(history.history['loss'], label='Train Loss')\n",
        "plt.plot(history.history['val_loss'], label='Val Loss')\n",
        "plt.title('Pérdida (Loss) vs Épocas')\n",
        "plt.legend()\n",
        "\n",
        "plt.subplot(1, 2, 2)\n",
        "plt.plot(history.history['accuracy'], label='Train Accuracy')\n",
        "plt.plot(history.history['val_accuracy'], label='Val Accuracy')\n",
        "plt.title('Precisión (Accuracy) vs Épocas')\n",
        "plt.legend()\n",
        "plt.show()"
    ]
})
nb6.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Predicciones en Testing set\n",
        "y_pred_probs = ann_model.predict(X_test_scaled)\n",
        "y_pred_enc = np.argmax(y_pred_probs, axis=1)\n",
        "\n",
        "y_pred = encoder.inverse_transform(y_pred_enc)  # Regresar de 0,1.. a 'fist'...\n",
        "\n",
        "print(\"Reporte de Clasificación ANN:\\n\")\n",
        "print(classification_report(y_test, y_pred, target_names=class_names))\n",
        "\n",
        "cm = confusion_matrix_report(y_test, y_pred, class_names)\n",
        "plot_confusion_matrix(cm, class_names, title='Matriz de Confusión - ANN')"
    ]
})
nb6.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Guardar el modelo y el escalador/encoder necesarios para inferencia en longitudional\n",
        "models_dir = Config.PROCESSED_DIR / 'models'\n",
        "os.makedirs(models_dir, exist_ok=True)\n",
        "\n",
        "ann_model.save(models_dir / 'best_ann_model.h5')\n",
        "joblib.dump(scaler, models_dir / 'ann_scaler.pkl')\n",
        "joblib.dump(encoder, models_dir / 'ann_encoder.pkl')\n",
        "\n",
        "print(f\"Modelo ANN y preprocesadores guardados en: {models_dir}\")"
    ]
})


# Generador master
for name, cells in [("04_modelo_knn.ipynb", nb4), ("05_modelo_svm.ipynb", nb5), ("06_modelo_ann.ipynb", nb6)]:
    nb = {"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
    with open(os.path.join(r"C:\Users\aseba\TIC\emg-classification-knn-svm-ann\notebooks", name), "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)

print("Notebooks 4, 5 and 6 generated successfully.")
