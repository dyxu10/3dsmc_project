import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree

# ---------- 1. Load target points (105 facial landmarks) ----------
target_path = Path("flame_mediapipe_landmarks.txt")
target_pts = []
with target_path.open() as f:
    for line in f:
        parts = line.split()
        if len(parts) == 4:                      # id  x  y  z
            target_pts.append([float(x) for x in parts[1:]])
target_pts = np.asarray(target_pts, dtype=np.float64)
assert target_pts.shape == (105, 3), f"Expect 105×3 but got {target_pts.shape}"

# ---------- 2. Load source points (vertices from the FLAME mesh) ----------
obj_path = Path("mesh/flame2023_random.obj")
verts = []
with obj_path.open() as f:
    for line in f:
        if line.startswith("v "):
            _, x, y, z = line.split()
            verts.append([float(x), float(y), float(z)])
verts = np.asarray(verts, dtype=np.float64)
print(f"Loaded {verts.shape[0]} vertices")

# ---------- 3. Build KDTree and query nearest neighbors ----------
tree = cKDTree(verts)
dists, indices = tree.query(target_pts, k=1)

# ---------- 4. Print and save the matching results ----------
print("\nLandmark ➜ Vertex mapping (index is 0-based in OBJ)")
with open("out/landmark_vertex_mapping.txt", "w") as fout:
    fout.write("lm_id  vert_id  dist\n")
    for i, (vid, d) in enumerate(zip(indices, dists), start=1):
        line = f"{i:3d}  {vid:5d}  {d:.6f}\n"
        print(line.rstrip())
        fout.write(line)

print("\n✔️  Saved mapping to out/landmark_vertex_mapping.txt")
