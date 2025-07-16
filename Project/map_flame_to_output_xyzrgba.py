import numpy as np
import open3d as o3d
from pathlib import Path
from scipy.spatial import cKDTree

# ======= Config ========
base_dir = Path(__file__).resolve().parent
mesh_off_path = base_dir / "out" / "00001_transform_onlyface.off"
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

MAX_DIST_THRESHOLD = 0.05  # ✅ 设定最大距离阈值，单位为米（可根据你模型单位调整）

for flame_id, f_vert in enumerate(flame_verts):
    k = 50
    dists, indices = tree.query(f_vert, k=k)
    if k == 1:
        dists = [dists]
        indices = [indices]

    found = False
    for dist, idx in zip(dists, indices):
        if not used[idx] and dist < MAX_DIST_THRESHOLD:  # ✅ 添加距离判断
            matched_flame_ids.append(flame_id)
            matched_output_ids.append(idx)
            matched_dists.append(dist)
            used[idx] = True
            found = True
            break

    if not found:
        print(f"⚠️  FLAME vertex {flame_id} could not find unused target point under threshold {MAX_DIST_THRESHOLD}")


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


# ========== 可视化增强版 ==========

# 红色：FLAME原始点
pcd_flame = o3d.geometry.PointCloud()
pcd_flame.points = o3d.utility.Vector3dVector(flame_verts)
pcd_flame.paint_uniform_color([1, 0, 0])  # 红色

# 绿色：匹配到的OFF点
pcd_matched = o3d.geometry.PointCloud()
pcd_matched.points = o3d.utility.Vector3dVector(mapped_xyz)
pcd_matched.paint_uniform_color([0, 1, 0])  # 强制绿色，避免被RGBA颜色影响

# 蓝线：连接线（可采样）
sample_rate = 1  # 每隔几个点显示一根线，避免太密
lines = [[i, i + len(flame_verts)] for i in range(0, len(matched_flame_ids), sample_rate)]
line_pts = np.vstack((flame_verts, mapped_xyz))
line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(line_pts)
line_set.lines = o3d.utility.Vector2iVector(lines)
line_set.colors = o3d.utility.Vector3dVector([[0, 0, 1]] * len(lines))  # 蓝色线

# 黄色：未匹配的 FLAME 点
unmatched_ids = set(range(len(flame_verts))) - set(matched_flame_ids)
if unmatched_ids:
    unmatched_pts = np.array([flame_verts[i] for i in unmatched_ids])
    pcd_unmatched = o3d.geometry.PointCloud()
    pcd_unmatched.points = o3d.utility.Vector3dVector(unmatched_pts)
    pcd_unmatched.paint_uniform_color([1, 1, 0])  # 黄色
    print(f"⚠️  可视化中显示 {len(unmatched_ids)} 个未匹配 FLAME 点（黄色）")
    geometries = [pcd_flame, pcd_matched, line_set, pcd_unmatched]
else:
    geometries = [pcd_flame, pcd_matched, line_set]

# 保存PLY文件（绿色OFF点）
ply_output_path = out_dir / "matched_output_points.ply"
o3d.io.write_point_cloud(str(ply_output_path), pcd_matched)
print(f"💾 可视化PLY文件已保存，可用MeshLab打开： {ply_output_path}")

# 展示
o3d.visualization.draw_geometries(geometries,
                                  window_name="FLAME to OFF Mapping (Enhanced)",
                                  point_show_normal=False)
