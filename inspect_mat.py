"""
Diagnóstico para depurar la navegación dentro de los archivos .mat
Corre este script en Colab ANTES del notebook 07 para verificar el loader.
"""
import scipy.io
import numpy as np
from pathlib import Path

mat_file = Path("data/longitudinal/usuarios/Mes0/user4/fist.mat")
raw = scipy.io.loadmat(str(mat_file))

print("=== Nivel 1: raw['reps'] ===")
reps = raw['reps']
print(f"type: {type(reps)}, shape: {reps.shape}, dtype: {reps.dtype}")

print("\n=== Nivel 2: reps['fist'] ===")
fist_outer = reps['fist']
print(f"type: {type(fist_outer)}, shape: {fist_outer.shape}, dtype: {fist_outer.dtype}")

print("\n=== Nivel 3: reps['fist'][0,0] ===")
fist_mid = reps['fist'][0, 0]
print(f"type: {type(fist_mid)}, shape: {getattr(fist_mid,'shape','N/A')}, dtype: {getattr(fist_mid,'dtype','N/A')}")

print("\n=== Nivel 4: acceder a 'data' ===")
data_val = fist_mid['data']
print(f"type: {type(data_val)}, shape: {getattr(data_val,'shape','N/A')}, dtype: {getattr(data_val,'dtype','N/A')}")

print("\n=== Nivel 5: data_val[0,0] ===")
d00 = data_val[0, 0]
print(f"type: {type(d00)}, shape: {getattr(d00,'shape','N/A')}, dtype: {getattr(d00,'dtype','N/A')}")

print("\n=== Nivel 6: primera repetición ===")
rep0 = data_val[0, 0]
# Puede ser un array (1,1) o un numpy.void
if hasattr(rep0, 'shape') and rep0.shape != () and len(rep0.shape) > 0:
    print(f"Es ndarray de shape {rep0.shape}. Haciendo flat[0]...")
    rep0 = rep0.flat[0]
    print(f"  Ahora: type={type(rep0)}, shape={getattr(rep0,'shape','N/A')}")

print(f"\nCampos disponibles: {rep0.dtype.names if hasattr(rep0,'dtype') and hasattr(rep0.dtype,'names') else 'N/A'}")

print("\n=== Accediendo a 'emg' ===")
emg_raw = rep0['emg']
print(f"  emg tipo: {type(emg_raw)}, shape: {getattr(emg_raw,'shape','N/A')}, dtype: {getattr(emg_raw,'dtype','N/A')}")

# Si sigue siendo object array, hacer flat[0]
if hasattr(emg_raw, 'dtype') and emg_raw.dtype == object:
    print("  emg es object array, haciendo flat[0]...")
    emg_raw = emg_raw.flat[0]
    print(f"  emg final: type={type(emg_raw)}, shape={getattr(emg_raw,'shape','N/A')}")

print(f"\n✅ Forma final de EMG: {np.array(emg_raw).shape}")
print(f"   Primeros valores: {np.array(emg_raw)[:3, :3]}")
