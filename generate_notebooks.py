import json
import os

nb2_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Notebook 2 \u2014 Exploraci\u00f3n de Datos (EDA)\n",
            "\n",
            "Este notebook carga el dataset parseado previamente en el Notebook 1 y permite:\n",
            "1. Ver estad\u00edsticas descriptivas del dataset.\n",
            "2. Graficar se\u00f1ales EMG crudas y preprocesadas.\n",
            "3. Verificar el impacto de los filtros aplicados a las se\u00f1ales."
        ]
    },
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
            "import matplotlib.pyplot as plt\n",
            "\n",
            "# Montaje y rutas de Drive igual que NB 1\n",
            "try:\n",
            "    from google.colab import drive\n",
            "    drive.mount('/content/drive', force_remount=True)\n",
            "    import glob\n",
            "    possible_paths = glob.glob('/content/drive/**/emg-classification-knn-svm-ann', recursive=True)\n",
            "    if len(possible_paths) > 0:\n",
            "        pc_paths = [p for p in possible_paths if 'Othercomputers' in p or 'Ordenadores' in p]\n",
            "        if pc_paths:\n",
            "            PROJECT_PATH = sorted(pc_paths, key=len)[0]\n",
            "        else:\n",
            "            PROJECT_PATH = sorted(possible_paths, key=len)[0]\n",
            "    else:\n",
            "        PROJECT_PATH = '/content/drive/Othercomputers/My PC/emg-classification-knn-svm-ann'\n",
            "    os.chdir(PROJECT_PATH)\n",
            "    if PROJECT_PATH not in sys.path:\n",
            "        sys.path.insert(0, PROJECT_PATH)\n",
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
            "from src.data_loader import get_signal_from_emg, segment_gesture\n",
            "from src.preprocessing import preprocess_pipeline, bandpass_filter\n",
            "from src.visualization import plot_signal\n",
            "\n",
            "# Cargar DataFrame del Notebook 1\n",
            "db_path = Config.PROCESSED_DIR / 'notebook1' / 'samples_full.pkl'\n",
            "df = pd.read_pickle(db_path)\n",
            "print(f\"Dataset cargado: {len(df)} muestras.\")\n",
            "df.head(3)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Resumen de Muestras por Gesto y Usuario\n",
            "resumen = df.groupby(['user_id', 'gesture_name']).size().unstack(fill_value=0)\n",
            "display(resumen)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Seleccionar una muestra de prueba al azar\n",
            "sample = df.sample(1).iloc[0]\n",
            "signal_raw = sample['signal_raw']\n",
            "gesture = sample['gesture_name']\n",
            "gt_idx = sample['ground_truth_index']\n",
            "\n",
            "print(f\"Visualizando se\u00f1al de clase '{gesture}'\")\n",
            "print(f\"Dimensiones crudas (incluyendo reposo): {signal_raw.shape}\")\n",
            "\n",
            "# 1. Plot Se\u00f1al cruda\n",
            "fig_raw = plot_signal(signal_raw, title=f\"Se\u00f1al EMG Cruda - {gesture}\")\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 2. Plot Se\u00f1al Recortada y Filtrada (Preprocesada)\n",
            "signal_segmented = segment_gesture(signal_raw, gt_idx)\n",
            "print(f\"Dimensiones de se\u00f1al recortada (solo acci\u00f3n): {signal_segmented.shape}\")\n",
            "\n",
            "# Aplicar Filtro\n",
            "signal_filtered = bandpass_filter(signal_segmented, fs=Config.SAMPLE_RATE, low=Config.LOW_CUTOFF, high=Config.HIGH_CUTOFF)\n",
            "\n",
            "fig_filtered = plot_signal(signal_filtered, title=f\"Se\u00f1al Recortada y Filtrada Bandpass (20-90Hz)\")\n",
            "plt.show()"
        ]
    }
]

nb3_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Notebook 3 \u2014 Extracci\u00f3n de Features\n",
            "\n",
            "Se aplican los scripts del m\u00f3dulo `src.feature_extraction` a todas las muestras, generando la matriz de caracter\u00edsticas `X` y el vector de etiquetas `y` que consumir\u00e1n los modelos."
        ]
    },
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
            "from tqdm import tqdm\n",
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
            "from src.data_loader import segment_gesture\n",
            "from src.preprocessing import preprocess_pipeline\n",
            "from src.feature_extraction import build_feature_matrix\n",
            "\n",
            "# Cargar DataFrame del Notebook 1\n",
            "db_path = Config.PROCESSED_DIR / 'notebook1' / 'samples_full.pkl'\n",
            "df = pd.read_pickle(db_path)\n",
            "print(f\"Muestras totales: {len(df)}\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Segmentaci\u00f3n y Preprocesamiento de todas las muestras\n",
            "tqdm.pandas(desc=\"Segmentando)\")\n",
            "\n",
            "def apply_prep(row):\n",
            "    sig = row['signal_raw']\n",
            "    gt = row['ground_truth_index']\n",
            "    seg_sig = segment_gesture(sig, gt)\n",
            "    # Filter and Rectify/Normalize\n",
            "    prep_sig = preprocess_pipeline(seg_sig)\n",
            "    return prep_sig\n",
            "\n",
            "print(\"Preprocesando las se\u00f1ales...\")\n",
            "df['signal_prep'] = df.progress_apply(apply_prep, axis=1)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Extracci\u00f3n de Features\n",
            "signals_list = df['signal_prep'].tolist()\n",
            "labels_list = df['gesture_name'].tolist()\n",
            "\n",
            "print(\"Extrayendo caracter\u00edsticas (Features de dominio del tiempo)...\")\n",
            "X, y = build_feature_matrix(signals_list, labels_list)\n",
            "\n",
            "print(f\"Shape de X (Features): {X.shape}\")\n",
            "print(f\"Shape de y (Labels): {y.shape}\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Guardado del conjunto procesado (X, y listos para ML)\n",
            "out_dir = Config.PROCESSED_DIR / 'features'\n",
            "os.makedirs(out_dir, exist_ok=True)\n",
            "\n",
            "np.savez_compressed(out_dir / 'emg_features_base.npz', X=X, y=y)\n",
            "print(f\"Guardado exitosamente en: {out_dir / 'emg_features_base.npz'}\")\n",
            "\n",
            "# Guardar labels e IDs adicionales si quieres cruzar informaci\u00f3n en train test split\n",
            "users_list = df['user_id'].tolist()\n",
            "splits_list = df['split'].tolist()\n",
            "np.savez_compressed(out_dir / 'emg_metadata_base.npz', y=y, users=users_list, splits=splits_list)\n",
            "print(\"Operaci\u00f3n Completada.\")"
        ]
    }
]

for name, cells in [("02_exploracion_datos.ipynb", nb2_cells), ("03_extraccion_features.ipynb", nb3_cells)]:
    nb = {
        "cells": cells,
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5
    }
    with open(os.path.join(r"C:\Users\aseba\TIC\emg-classification-knn-svm-ann\notebooks", name), 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

print("Notebooks 2 and 3 generated.")
