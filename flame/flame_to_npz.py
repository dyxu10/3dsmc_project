import numpy as np

# 加载 FLAME 模型文件
npz_path = 'flame2023_no_jaw.npz'
data = np.load(npz_path)

# 获取 shapedirs 的维度
shapedirs = data['shapedirs']
print("shapedirs shape:", shapedirs.shape)

# 获取 shape basis 的数量（即第三维度）
num_shape_params = shapedirs.shape[2]
print("Number of shape parameters (β):", num_shape_params)