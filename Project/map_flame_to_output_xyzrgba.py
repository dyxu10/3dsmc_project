import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree

# ======= Config ========
base_dir = Path(__file__).resolve().parent
mesh_off_path = base_dir / "out" / "output_1_transform.off"
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

# ======= Build unique 1-to-1 mapping FLAME ➜ OFF ========
print("🚀 Fast matching with reuse-aware KDTree search...")

matched_flame_ids = []
matched_output_ids = []
matched_dists = []
tree = cKDTree(off_xyz)
used = np.zeros(len(off_xyz), dtype=bool)

for flame_id, f_vert in enumerate(flame_verts):
    k = 10000
    dists, indices = tree.query(f_vert, k=k)
    if k == 1:
        dists = [dists]
        indices = [indices]

    found = False
    for dist, idx in zip(dists, indices):
        if not used[idx]:
            matched_flame_ids.append(flame_id)
            matched_output_ids.append(idx)
            matched_dists.append(dist)
            used[idx] = True
            found = True
            break

    if not found:
        print(f"⚠️  FLAME vertex {flame_id} could not find unused target point among top-{k} nearest.")

print(f"✅ Matched {len(matched_flame_ids)} FLAME vertices with unique OFF points.")


# ======= Save mapping TXT ========
mapping_txt = out_dir / "flame2output_vertex_mapping_unique.txt"
with mapping_txt.open("w") as fout:
    fout.write("flame_vert_id  off_vert_id  dist\n")
    for flame_id, off_id, d in zip(matched_flame_ids, matched_output_ids, matched_dists):
        fout.write(f"{flame_id:7d}     {off_id:7d}   {d:.6f}\n")
print("✔️  Mapping saved to", mapping_txt)

# ======= Save points-only TXT (no color) ========
mapped_xyz = off_xyz[matched_output_ids]
np.savetxt(out_dir / "flame2output_points_xyz_unique.txt",
           mapped_xyz,
           fmt="%.6f")

# ======= Save xyz+rgba TXT ========
mapped_rgba = off_rgba[matched_output_ids]
np.savetxt(out_dir / "flame2output_points_xyzrgba_unique.txt",
           np.hstack([mapped_xyz, mapped_rgba]),
           fmt="%.6f %.6f %.6f %d %d %d %d")

np.savetxt(out_dir / "matched_flame_indices.txt", matched_flame_ids, fmt="%d")


print("✅ Saved uniquely matched xyz and xyzrgba point lists.")