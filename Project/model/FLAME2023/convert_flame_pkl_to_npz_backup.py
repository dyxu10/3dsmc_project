#!/usr/bin/env python
"""
convert_flame_pkl_to_npz.py
Convert a FLAME *.pkl (saved with chumpy) to *.npz, without installing chumpy.
Works with modern NumPy (>=1.24) – fixes object-dtype view error.
"""
import sys, types, pickle, numpy as np
from pathlib import Path

# ---------- 1. fake the minimal chumpy API ---------- #
class _DummyCh(np.ndarray):
    def __new__(cls, data=None):
        arr = np.asarray(data if data is not None else 0)
        if arr.dtype == object:
            return arr
        return arr.view(cls)

    def __array_finalize__(self, obj):
        pass

    def __reduce__(self):
        return (_DummyCh, (np.asarray(self),))

    def __setstate__(self, state):
        if isinstance(state, dict):
            for k, v in state.items():
                setattr(self, k, v)
        elif isinstance(state, (tuple, list)) and len(state) == 4:
            self._arr = state[0]
        else:
            pass

chumpy          = types.ModuleType("chumpy")
chumpy.array    = np.array
chumpy.ndarray  = np.ndarray
chumpy.Ch       = _DummyCh
sys.modules["chumpy"] = chumpy

chumpy_ch       = types.ModuleType("chumpy.ch")
chumpy_ch.Ch    = _DummyCh
sys.modules["chumpy.ch"] = chumpy_ch

# ---------- 2. restore deprecated NumPy aliases ---------- #
for alias, typ in dict(bool=bool, int=int, float=float,
                       complex=complex, object=object,
                       str=str, unicode=str).items():
    if not hasattr(np, alias):
        setattr(np, alias, typ)

# ---------- 3. paths ---------- #
input_pkl  = Path("flame2023_no_jaw.pkl")
output_npz = input_pkl.with_suffix(".npz")
if not input_pkl.exists():
    sys.exit(f"❌  Input not found: {input_pkl}")

# ---------- 4. convert (no rename) ---------- #
with input_pkl.open("rb") as fh:
    flame_dict = pickle.load(fh, encoding="latin1")

np.savez(output_npz, **flame_dict)

print("Keys in saved npz file:")
for key in flame_dict.keys():
    print("  ", key)

print(f"✅  Converted → {output_npz}")
