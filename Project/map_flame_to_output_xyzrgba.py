import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree

# ======= Config ========
base_dir = Path(__file__).resolve().parent
mesh_off_path = base_dir / "out" / "output_transformed_1.off"
flame_obj_path = base_dir / "model" / "mesh" / "flame2023_random.obj"
out_dir = base_dir / "out"
out_dir.mkdir(parents=True, exist_ok=True)

# ======= Load COFF (with color) ========
def load_off_xyzrgba(path: Path):
    with path.open("r") as f:
        header = f.readline().strip()
        if header not in {"OFF", "COFF"}:
            raise ValueError(f"{path} is not OFF/COFF")
        n_verts = int(f.readline().split()[0])
        xyz = []
        rgba = []
        for _ in range(n_verts):
            parts = f.readline().split()
            xyz.append(list(map(float, parts[:3])))
            rgba.append(list(map(int, parts[3:7])) if len(parts) >= 7 else [0, 0, 0, 255])
    return np.asarray(xyz), np.asarray(rgba)

print("🔹 Loading OFF mesh:", mesh_off_path)
off_xyz, off_rgba = load_off_xyzrgba(mesh_off_path)
print("   →", off_xyz.shape[0], "vertices")

# ======= Load FLAME OBJ ========
print("🔹 Loading FLAME OBJ:", flame_obj_path)
flame_verts = []
with flame_obj_path.open() as f:
    for line in f:
        if line.startswith("v "):
            _, x, y, z = line.split()
            flame_verts.append([float(x), float(y), float(z)])
flame_verts = np.asarray(flame_verts, dtype=np.float64)
print("   →", flame_verts.shape[0], "vertices")

# ======= Build mapping FLAME ➜ OFF ========
tree = cKDTree(off_xyz)
dists, indices = tree.query(flame_verts, k=1)

# ======= Save mapping TXT ========
mapping_txt = out_dir / "flame2output_vertex_mapping.txt"
with mapping_txt.open("w") as fout:
    fout.write("flame_vert_id  off_vert_id  dist\n")
    for flame_id, (off_id, d) in enumerate(zip(indices, dists)):
        fout.write(f"{flame_id:7d}     {off_id:7d}   {d:.6f}\n")
print("✔️  Mapping saved to", mapping_txt)

# ======= Save points-only TXT (no color) ========
mapped_xyz = off_xyz[indices]
np.savetxt(out_dir / "flame2output_points_xyz.txt",
           mapped_xyz,
           fmt="%.6f")

# ======= Save xyz+rgba TXT ========
mapped_rgba = off_rgba[indices]
np.savetxt(out_dir / "flame2output_points_xyzrgba.txt",
           np.hstack([mapped_xyz, mapped_rgba]),
           fmt="%.6f %.6f %.6f %d %d %d %d")

print("✅ Saved xyz and xyzrgba point lists.")

