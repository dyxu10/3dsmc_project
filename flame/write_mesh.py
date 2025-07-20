import pickle
import numpy as np
import os

# 路径配置
model_path = "./model/FLAME2023/flame2023.pkl"
shape_param_path = "test_betas.txt"
output_obj_path = "newshape.obj"

# 加载 FLAME 模型
with open(model_path, "rb") as f:
    model = pickle.load(f, encoding='latin1')

# 提取模板和 shape basis
v_template = model['v_template']        # [N, 3]
shapedirs = model['shapedirs']          # [N, 3, 300]
faces = model['f']                      # [F, 3]

# ✅ 加载优化后的 shape 参数（必须是 1D 300 维）
shape_params = np.loadtxt(shape_param_path)  # shape: [300]
assert shape_params.shape[0] == shapedirs.shape[-1], \
    f"Shape param dim {shape_params.shape[0]} doesn't match shapedirs {shapedirs.shape[-1]}"

# 生成变形后的人脸 mesh
# shapedirs: [N, 3, 300] → reshape to [N, 3]
blend_shape = np.tensordot(shapedirs, shape_params, axes=[2, 0])  # [N, 3]
v_shaped = v_template + blend_shape  # [N, 3]

# 写入 .obj
def save_obj(path, vertices, faces):
    with open(path, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

save_obj(output_obj_path, v_shaped, faces)
print(f"✅ Exported: {output_obj_path}")