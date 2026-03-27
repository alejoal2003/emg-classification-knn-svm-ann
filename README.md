# Hand Gesture Recognition (HGR) using sEMG

Este proyecto define un pipeline completo para la clasificación de gestos de la mano utilizando señales de Electromiografía de Superficie (sEMG). Implementa tres modelos de Machine Learning (kNN, SVM y Redes Neuronales Artificiales - ANN) para procesar datos de múltiples usuarios procedentes de dispositivos como la pulsera Myo.

---

## 📁 Estructura del Proyecto

```
emg-classification-knn-svm-ann/
├── data/                       # No subido a GitHub (Ignorado en .gitignore)
│   ├── trainingJSON/           # -> COLOCAR AQUÍ LOS JSON DE ENTRENAMIENTO
│   ├── testingJSON/            # -> COLOCAR AQUÍ LOS JSON DE PRUEBA (TESTING)
│   └── processed/              # Carpeta generada automáticamente
│       ├── notebook1/          # DataFrames generados en NB 01
│       ├── features/           # Vectores de características generados en NB 03
│       └── models/             # Modelos entrenados exportados (NB 04, 05, 06)
├── notebooks/                  # Flujo de ejecución principal
│   ├── 01_carga_parsing_json.ipynb
│   ├── 02_exploracion_datos.ipynb
│   ├── 03_extraccion_features.ipynb
│   ├── 04_modelo_knn.ipynb
│   ├── 05_modelo_svm.ipynb
│   └── 06_modelo_ann.ipynb
├── src/                        # Código base del pipeline (módulos Python)
│   ├── config.py               # Rutas estáticas y constantes genéricas
│   ├── data_loader.py          # Lógica para leer los JSON y estructurarlos
│   ├── preprocessing.py        # Filtros digitales y recorte de señales sEMG
│   ├── feature_extraction.py   # Variables dominicales (tiempo y frecuencia)
│   ├── evaluation.py           # Métricas de validación de los modelos
│   └── visualization.py        # Soporte para gráficos (señales, matrices)
├── rewrite_notebooks.py        # Script genérico para regerar los notebooks base
└── README.md                   # Esta documentación
```

---

## 💾 Dónde colocar y cómo procesar los datos

Para que el proyecto funcione en un nuevo entorno local o nube (por ejemplo, Google Drive ligado con Google Colab):

1. **Obtener Base de Datos:** Los archivos crudos consisten en formatos `.json` provenientes del dispositivo de 8 canales.
2. **Ubicación:** 
   - Coloca todos los archivos JSON de entrenamiento (`user01_train.json`, `user02_train.json`, etc.) dentro del directorio **`data/trainingJSON/`**.
   - Coloca los archivos JSON de testing correspondientes dentro del directorio **`data/testingJSON/`**.
3. **Flujo de Notebooks:**
   - Una vez colocados los datasets, debes ejecutar el **Notebook 01** (`01_carga_parsing_json.ipynb`). Este convertirá todos los JSONs en un único gran `DataFrame` exportable (`samples_full.pkl`) dentro de `data/processed/notebook1/`.
   - Luego, usa el **Notebook 03** para extraer los *Features*. Esto usará los métodos en *feature_extraction.py* (MAV, RMS, WL, etc., tanto en dominios de tiempo como frecuencia).
   - Ahora puedes entrenar usando cualquiera de los notebooks **04 (kNN)**, **05 (SVM)**, o **06 (ANN)** independientemente, puesto que todos tomarán sus datos de las características ya extraídas dentro de `data/processed/features/`.

---

## 🛠️ Ejecución en Google Colab

El entorno está preparado para integrarse perfectamente a Google Colab si la carpeta raíz del proyecto se subió sincronizada usando el cliente de escritorio de *Google Drive* a la ruta habitual:
- `MyDrive/.../emg-classification-knn-svm-ann` o en su defecto
- `Othercomputers/.../emg-classification-knn-svm-ann`.

La directiva que hace que se auto-posicione internamente está en todos los cuadernos bajo el código inicial de `drive.mount()`.

**GPU Availability:** 
- `kNN`, `SVM`: Usan primariamente la CPU. Se recomienda paralelización mediante `n_jobs=-1`.
- `ANN`: Funciona exponencialmente más veloz usando GPUs con CUDA y TensorFlow activado (Ajustar *Change runtime type -> T4 GPU* en Colab antes del entrenamiento).
