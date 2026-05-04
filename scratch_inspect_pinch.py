import scipy.io
from pathlib import Path

mat_path = r"c:\Users\aseba\TIC\emg-classification-knn-svm-ann\data\longitudinal\usuarios\Mes2\user14\pinch.mat"
print(f"File exists: {Path(mat_path).exists()}")

if Path(mat_path).exists():
    try:
        raw = scipy.io.loadmat(mat_path, simplify_cells=True)
        gesture_data = raw['reps']['pinch']['data']
        print(f"Type of gesture_data: {type(gesture_data)}")
        if isinstance(gesture_data, list):
            print(f"Len of gesture_data: {len(gesture_data)}")
            print(f"Type of first element: {type(gesture_data[0])}")
        import numpy as np
        if isinstance(gesture_data, np.ndarray):
            print(f"Shape of gesture_data: {gesture_data.shape}")
            print(f"Type of first element: {type(gesture_data[0])}")
            if gesture_data.shape[0] > 0:
                print(f"Keys/fields of first element: {getattr(gesture_data[0], 'dtype', 'Not a structured array')}")
    except Exception as e:
        print(f"Error: {e}")
