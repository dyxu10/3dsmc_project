import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree

def load_off_vertices(path: Path) -> np.ndarray:
    with path.open("r") as f:
        header = f.readline().strip()
        if header not in {"OFF", "COFF"}:
            raise ValueError(f"{path} is not a valid OFF/COFF file")
        n_verts = int(f.readline().split()[0])
        verts = [list(map(float, f.readline().split()[:3])) for _ in range(n_verts)]
    return np.asarray(verts, dtype=np.float64)

base_dir      = Path(__file__).resolve().parent
mesh_off_path = base_dir / "out" / "output_transformed_1.off"          # OK
flame_obj_path = base_dir / "model" / "mesh" / "flame2023_random.obj"  # ← 修正
# 如 FLAME 在 files/flame/mesh，则改成
# flame_obj_path = base_dir.parent / "flame" / "mesh" / "flame2023_random.obj"
out_txt_path  = base_dir / "out" / "flame2output_vertex_mapping.txt"

print("🔹 Loading OFF mesh:", mesh_off_path)
off_verts = load_off_vertices(mesh_off_path)
print("   →", off_verts.shape[0], "vertices")

print("🔹 Loading FLAME OBJ:", flame_obj_path)
flame_verts = []
with flame_obj_path.open() as f:
    for line in f:
        if line.startswith("v "):
            _, x, y, z = line.split()
            flame_verts.append([float(x), float(y), float(z)])
flame_verts = np.asarray(flame_verts, dtype=np.float64)
print("   →", flame_verts.shape[0], "vertices")


tree = cKDTree(off_verts)
dists, indices = tree.query(flame_verts, k=1)


out_txt_path.parent.mkdir(parents=True, exist_ok=True)
with out_txt_path.open("w") as fout:
    fout.write("flame_vert_id  off_vert_id  dist\n")
    for flame_id, (off_id, d) in enumerate(zip(indices, dists)):
        fout.write(f"{flame_id:7d}     {off_id:7d}   {d:.6f}\n")

print("✔️  Mapping saved to", out_txt_path)