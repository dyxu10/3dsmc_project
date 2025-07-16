import numpy as np
import open3d as o3d
import os
import matplotlib.pyplot as plt

# ========== 文件路径 ==========
base_dir = "../optimizer"
flame_path = os.path.join(base_dir, "flame.txt")
matched_path = os.path.join(base_dir, "matched.txt")

# ========== 读取点云 ==========
flame = np.loadtxt(flame_path)
matched = np.loadtxt(matched_path)

# ========== 创建点云对象 ==========
pcd_flame = o3d.geometry.PointCloud()
pcd_flame.points = o3d.utility.Vector3dVector(flame)
pcd_flame.paint_uniform_color([1, 0, 0])  # 红色

pcd_matched = o3d.geometry.PointCloud()
pcd_matched.points = o3d.utility.Vector3dVector(matched)

# ========== 用距离上色 ==========
distances = np.linalg.norm(flame - matched, axis=1)
max_d = distances.max()
colors = plt.get_cmap("jet")(distances / max_d)[:, :3]  # RGB
pcd_matched.colors = o3d.utility.Vector3dVector(colors)

# ========== 添加连线 ==========
all_points = np.vstack((flame, matched))  # 合并为 LineSet 点坐标
lines = [[i, i + len(flame)] for i in range(len(flame))]  # 每条线连接 flame[i] -> matched[i]
line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(all_points),
    lines=o3d.utility.Vector2iVector(lines)
)
line_set.colors = o3d.utility.Vector3dVector([[0, 0, 1] for _ in lines])  # 蓝色线段

# ========== 可视化 ==========
o3d.visualization.draw_geometries([pcd_flame, pcd_matched, line_set])
