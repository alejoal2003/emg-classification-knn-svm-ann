import json
import os

nb_path = r'C:\Users\aseba\TIC\emg-classification-knn-svm-ann\notebooks\01_carga_parsing_json.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_source_str = """# ==============================
# Montaje de Drive (Solo para Colab)
# ==============================
try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)
    IN_COLAB = True

    # Rutas comunes por donde Colab monta carpetas sincronizadas de PC
    import glob
    possible_paths = glob.glob('/content/drive/**/emg-classification-knn-svm-ann', recursive=True)

    if len(possible_paths) > 0:
        print(f"Búsqueda automática encontró {len(possible_paths)} carpeta(s):")
        for p in possible_paths: print(f" - {p}")
        
        # Priorizar la carpeta de Google Drive Desktop ('Othercomputers' o 'Ordenadores')
        pc_paths = [p for p in possible_paths if 'Othercomputers' in p or 'Ordenadores' in p]
        if pc_paths:
            PROJECT_PATH = sorted(pc_paths, key=len)[0] # Tomar la más corta que sea de PC
        else:
            PROJECT_PATH = sorted(possible_paths, key=len)[0]
        print(f"\\nUsando carpeta de proyecto activa: {PROJECT_PATH}")
    else:
        # Fallback manual en caso de que glob falle
        print("\\nATENCIÓN: Búsqueda automática falló. Intentando rutas manuales:")
        path_pc = '/content/drive/Othercomputers/My PC/emg-classification-knn-svm-ann'
        path_mi_pc = '/content/drive/Othercomputers/Mi PC/emg-classification-knn-svm-ann'
        path_drive = '/content/drive/MyDrive/emg-classification-knn-svm-ann'
        path_ordenadores = '/content/drive/MyDrive/Ordenadores/My PC/emg-classification-knn-svm-ann'

        if os.path.exists(path_pc): PROJECT_PATH = path_pc
        elif os.path.exists(path_mi_pc): PROJECT_PATH = path_mi_pc
        elif os.path.exists(path_drive): PROJECT_PATH = path_drive
        elif os.path.exists(path_ordenadores): PROJECT_PATH = path_ordenadores
        else:
            raise FileNotFoundError("No se encontró la carpeta en Drive.")

    os.chdir(PROJECT_PATH)
    if PROJECT_PATH not in sys.path:
        sys.path.append(PROJECT_PATH)

except Exception as e:
    print(f"Excepción en Colab: {e}")
    IN_COLAB = False
    # Si estás en local
    PROJECT_PATH = os.getcwd()
    if 'notebooks' in PROJECT_PATH:
        PROJECT_PATH = os.path.abspath('..')
        os.chdir(PROJECT_PATH)
    if PROJECT_PATH not in sys.path:
        sys.path.append(PROJECT_PATH)

print(f'\\nIN_COLAB = {IN_COLAB}\\nDirectorio de trabajo actual: {os.getcwd()}')"""

new_source_list = [line + '\n' for line in new_source_str.split('\n')]
new_source_list[-1] = new_source_list[-1].strip('\n')

for cell in nb['cells']:
    if cell.get('cell_type') == 'code':
        # Buscamos la celda de montaje de Drive
        src_text = "".join(cell.get('source', []))
        if "google.colab import drive" in src_text and "Montaje" in src_text:
            cell['source'] = new_source_list
            cell['outputs'] = []  # clear outputs so it's clean
            cell['execution_count'] = None

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook actualizado con éxito.")
