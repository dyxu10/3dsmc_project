import pickle
import chumpy
import numpy as np
import os

file_name = "flame2023_no_jaw"

# 获取当前脚本的路径
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "model", "FLAME2023", file_name + ".pkl")
txt_output_path = os.path.join(script_dir, "mesh", file_name + ".txt")

# 加载模型
with open(model_path, "rb") as f:
    model = pickle.load(f, encoding='latin1')

v_template = model['v_template']
# 如果想加入形状变形可取消注释以下两行
# shapedirs = model['shapedirs']
# betas = chumpy.random.randn(400)
# v_shaped = v_template + shapedirs.dot(betas)
v_shaped = v_template  # 这里只导出 mean face

# 保存顶点到 txt
with open(txt_output_path, "w") as f:
    for v in v_shaped:
        f.write(f"{v[0]} {v[1]} {v[2]}\n")

print(f"Exported vertices to {file_name}.txt")
