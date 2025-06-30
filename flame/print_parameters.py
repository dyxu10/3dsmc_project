import pickle
import chumpy
import numpy as np
from scipy.sparse import spmatrix
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
# Replace with your actual .pkl file path
pkl_path = script_dir + "/model/FLAME2023/flame2023.pkl"

with open(pkl_path, "rb") as f:
    data = pickle.load(f, encoding='latin1')  # encoding='latin1' helps with Python 2 -> 3 compatibility

# Now, `data` holds whatever object was pickled — often a dict or custom object
# print(type(data))       # Check the type of the loaded object
# print(data.keys())      # If it's a dict, list the keys

# Example: inspect shapedirs if present
# if 'shapedirs' in data:
#     print("shapedirs shape:", data['shapedirs'].shape)


print(type(data["f"]))
print(data["f"].shape)

print(f"📂 Loaded: {pkl_path}")
print("🔑 Keys:", list(data.keys()))
print("\n🔍 Details per key:\n")

for key, value in data.items():
    print(f"▶ {key}:")

    if isinstance(value, np.ndarray):
        print(f"  type: np.ndarray\n  shape: {value.shape}\n  dtype: {value.dtype}")

    elif isinstance(value, chumpy.Ch):
        print(f"  type: chumpy.Ch (symbolic tensor)\n  shape: {value.shape}")

    elif isinstance(value, spmatrix):
        print(f"  type: scipy.sparse matrix\n  shape: {value.shape}")

    elif isinstance(value, list):
        print(f"  type: list\n  length: {len(value)}")
        if len(value) > 0:
            print(f"  first element type: {type(value[0])}")
            if isinstance(value[0], np.ndarray):
                print(f"  first element shape: {value[0].shape}")

    elif isinstance(value, dict):
        print(f"  type: dict\n  keys: {list(value.keys())}")

    elif isinstance(value, str):
        print(f"  type: str\n  value: '{value}'")

    else:
        print(f"  type: {type(value)}")

