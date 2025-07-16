import numpy as np
import open3d as o3d
import os
import matplotlib.pyplot as plt

# ========== File paths ==========
base_dir = "../optimizer"
flame_path = os.path.join(base_dir, "flame.txt")
matched_path = os.path.join(base_dir, "matched.txt")

# ========== Load point clouds ==========
flame = np.loadtxt(flame_path)
matched = np.loadtxt(matched_path)

# ========== Create Open3D point clouds ==========
pcd_flame = o3d.geometry.PointCloud()
pcd_flame.points = o3d.utility.Vector3dVector(flame)
pcd_flame.paint_uniform_color([1, 0, 0])  # 红色

pcd_matched = o3d.geometry.PointCloud()
pcd_matched.points = o3d.utility.Vector3dVector(matched)

# ========== Color matched points by distance ==========
distances = np.linalg.norm(flame - matched, axis=1)
max_d = distances.max()
colors = plt.get_cmap("jet")(distances / max_d)[:, :3]  # RGB
pcd_matched.colors = o3d.utility.Vector3dVector(colors)

# ========== Add connecting lines between flame and matched ==========
all_points = np.vstack((flame, matched))  # Combine all points for LineSet
lines = [[i, i + len(flame)] for i in range(len(flame))]  # Line from flame[i] to matched[i]
line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(all_points),
    lines=o3d.utility.Vector2iVector(lines)
)
line_set.colors = o3d.utility.Vector3dVector([[0, 0, 1] for _ in lines])  # Blue lines

# ========== Visualize ==========
o3d.visualization.draw_geometries([pcd_flame, pcd_matched, line_set])
