import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any

from src.config import Config

def natural_channel_sort(channel_names: List[str]) -> List[str]:
    """Ordena canales tipo ch1,ch2,... o channel1,... de forma natural."""
    def key_fn(x: str):
        digits = ''.join([c for c in x if c.isdigit()])
        return (''.join([c for c in x if not c.isdigit()]), int(digits) if digits else 0)
    return sorted(channel_names, key=key_fn)

def emg_dict_to_array(emg_dict: Dict[str, Any]) -> np.ndarray:
    """Convierte dict de canales a array numpy de shape [T, C]."""
    channel_names = natural_channel_sort(list(emg_dict.keys()))
    channels = [np.asarray(emg_dict[ch], dtype=np.float32) for ch in channel_names]
    # Asegurarnos de que todos los canales tienen la misma longitud (a veces varía por 1 muestra)
    min_len = min(len(ch) for ch in channels)
    channels = [ch[:min_len] for ch in channels]
    x = np.stack(channels, axis=1)
    return x

def parse_samples_block(samples_block: Dict[str, Any], user_id: str, split_name: str) -> List[Dict[str, Any]]:
    """Parsea un bloque de trainingSamples o testingSamples del JSON."""
    rows = []
    
    # Filtrar solo los gestos que nos interesan
    valid_gestures = Config.GESTURES
    if Config.INCLUDE_NOGESTURE:
        valid_gestures.append('noGesture')
        
    for sample_key, sample in samples_block.items():
        gesture = sample.get('gestureName', None)
        
        # Ignorar gestos que no están en la lista (ej. noGesture si no lo queremos)
        if gesture not in valid_gestures:
            continue
            
        emg = sample.get('emg', {})
        x_signal = emg_dict_to_array(emg)

        gt = sample.get('groundTruth', None)
        if gt is not None:
            gt = np.asarray(gt, dtype=np.int8)

        gt_idx = sample.get('groundTruthIndex', None)
        if gt_idx is not None:
            gt_idx = np.asarray(gt_idx, dtype=np.int32)

        rows.append({
            'user_id': user_id,
            'split': split_name,
            'sample_key': sample_key,
            'gesture_name': gesture,
            'signal_len': int(x_signal.shape[0]),
            'n_channels': int(x_signal.shape[1]),
            'signal_raw': x_signal,
            'ground_truth': gt,
            'ground_truth_index': gt_idx
        })
    return rows

def load_user_json(json_path: Path) -> List[Dict[str, Any]]:
    """Carga y parsea un archivo JSON de usuario completo."""
    user_id = json_path.stem
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    rows = []
    train_block = data.get('trainingSamples', {})
    test_block = data.get('testingSamples', {})

    rows.extend(parse_samples_block(train_block, user_id=user_id, split_name='train_samples'))
    rows.extend(parse_samples_block(test_block, user_id=user_id, split_name='test_samples'))
    
    return rows

def load_all_users(json_dir: str, max_users: int = None) -> pd.DataFrame:
    """
    Busca archivos JSON de usuarios en la ruta dada y los carga en un DataFrame Pandas.
    Args:
        json_dir: Ruta al directorio (ej. config.TRAIN_JSON_DIR)
        max_users: Limita el número de usuarios a cargar (útil para pruebas)
    Returns:
        DataFrame con todos los datos.
    """
    root = Path(json_dir)
    # Patrón común del EMG-EPN-612: carpetas userXXX/userXXX.json
    files = sorted(root.glob('user*/user*.json'))
    
    # Si no se encuentra con ese patrón, intentar patrón directo
    if len(files) == 0:
        files = sorted(root.glob('*.json'))
        
    if max_users is not None:
        files = files[:max_users]
        
    all_rows = []
    
    # Intenta usar tqdm si está disponible, sino for loop normal
    try:
        from tqdm.auto import tqdm
        iterator = tqdm(files, desc=f'Parsing JSON from {root.name}')
    except ImportError:
        iterator = files
        print(f"Parsing {len(files)} files...")
        
    for p in iterator:
        all_rows.extend(load_user_json(p))
        
    df = pd.DataFrame(all_rows)
    return df
