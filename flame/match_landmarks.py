import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 分别画出dataset（pepnet-98个点）和flame（105个点）的landmarks并编号，用于手动匹配

script_dir = os.path.dirname(os.path.abspath(__file__))
flame_path = os.path.join(script_dir, "flame_mediapipe_landmarks.txt")
pn_path    = os.path.join(script_dir, "data/dataset_landmarks_3d/landmarks3D_1_woF.txt")

# 读取数据
flame_data = np.loadtxt(flame_path)  # shape (105, 4): idx x y z
pn_data    = np.loadtxt(pn_path)     # shape (98, 3): x y z

# 拆分 FLAME 数据
flame_idx = flame_data[:, 0].astype(int)
flame_pts = flame_data[:, 1:4]

# 给我们自己的landmarks加idx
pn_pts = pn_data  # 直接就是 (98,3)
pn_idx = np.arange(pn_pts.shape[0])  # 生成 0–97


# === FLAME Figure ===
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(*flame_pts.T, c='C0', alpha=0.6)
for i,p in zip(flame_idx, flame_pts):
    ax.text(*p, str(i), fontsize=8)
ax.set_title('FLAME Landmarks')
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
plt.show()

# === MediaPipe Figure ===
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(*pn_pts.T, c='C1', alpha=0.6)
for i,p in zip(pn_idx, pn_pts):
    ax.text(*p, str(i), fontsize=8)
ax.set_title('PipNet Landmarks')
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
plt.show()
