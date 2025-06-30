import os
import numpy as np
# 手动匹配后生成新的两个txt的landmarks文件（在matching文件夹下）
# pairs 列表
pairs = [
    (20, 37),(17, 36),(19, 35),(15, 34),
    (14, 38),(16, 39),(12, 40),(13, 41),
    (10, 42),(7, 43),(9, 44),(5, 45),
    (4, 50),(6, 49),(2, 48),(3, 47),(1, 46),
    (47, 62),(45, 63),(52, 61),(41, 66),
    (23, 68),(31, 70),(25, 74),(22, 72),
    (54, 52),(56, 53),(58, 54),(63, 58),(65, 59),
    (73, 76),(86, 77),(72, 78),(71, 79),(70, 80),
    (83, 87),(85, 86),(78, 85),(75, 89),(80, 95),
    (95, 91),(97, 95),
]


script_dir = os.path.dirname(os.path.abspath(__file__))
flame_path = os.path.join(script_dir, "flame_mediapipe_landmarks.txt")
pip_path   = os.path.join(script_dir, "data/dataset_landmarks_3d/landmarks3D_1_woF.txt")

# 读入数据
flame_data = np.loadtxt(flame_path)  # shape (105,4): idx x y z
pip_data   = np.loadtxt(pip_path)    # shape (98,3): x y z

# 提取坐标数组
# FLAME 的 idx 从 1 开始，所以要 f-1
flame_pts = { int(row[0]) : row[1:4] for row in flame_data }
pip_pts   = { i : pip_data[i] for i in range(pip_data.shape[0]) }

# 按 pairs 顺序生成对齐后的列表
flame_aligned = []
pip_aligned   = []
for f_idx, p_idx in pairs:
    flame_aligned.append(flame_pts[f_idx])
    pip_aligned.append(  pip_pts[p_idx])

flame_aligned = np.vstack(flame_aligned)  # (M,3)
pip_aligned   = np.vstack(pip_aligned)    # (M,3)

# 写出新文件（每行：x y z）
out_flame = os.path.join(script_dir, "matching/flame_matching.txt")
out_pip   = os.path.join(script_dir, "matching/pip_matching.txt")

np.savetxt(out_flame, flame_aligned, fmt="%.6f %.6f %.6f")
np.savetxt(out_pip,   pip_aligned,   fmt="%.6f %.6f %.6f")

print(f"Saved {flame_aligned.shape[0]} matched points to:")
print("  ", out_flame)
print("  ", out_pip)
