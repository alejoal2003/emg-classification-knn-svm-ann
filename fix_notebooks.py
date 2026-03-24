import json
import os

repo = r"c:\Users\aseba\TIC\emg-classification-knn-svm-ann\notebooks"

# Fix NB 02
nb2_path = os.path.join(repo, "02_exploracion_datos.ipynb")
with open(nb2_path, "r", encoding="utf-8") as f:
    nb2 = json.load(f)

for cell in nb2['cells']:
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            if "get_signal_from_emg" in line:
                continue # drop line "from src.data_loader import get_signal_from_emg, segment_gesture\n"
            if "from src.preprocessing import preprocess_pipeline, bandpass_filter" in line:
                line = "from src.preprocessing import preprocess_pipeline, bandpass_filter, segment_gesture\n"
            new_source.append(line)
        cell['source'] = new_source

with open(nb2_path, "w", encoding="utf-8") as f:
    json.dump(nb2, f, indent=1)

# Fix NB 03
nb3_path = os.path.join(repo, "03_extraccion_features.ipynb")
with open(nb3_path, "r", encoding="utf-8") as f:
    nb3 = json.load(f)

for cell in nb3['cells']:
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            if "from src.data_loader import segment_gesture" in line:
                continue # drop
            if "from src.preprocessing import preprocess_pipeline" in line:
                line = "from src.preprocessing import preprocess_pipeline, segment_gesture\n"
            new_source.append(line)
        cell['source'] = new_source

with open(nb3_path, "w", encoding="utf-8") as f:
    json.dump(nb3, f, indent=1)

print("Notebooks 02 and 03 fixed!")
