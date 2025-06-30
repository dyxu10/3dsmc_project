import pickle
import chumpy
import chumpy.ch_random
import numpy as np
import os

file_name = "flame2023_no_jaw"

# 获取当前脚本的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "model", "FLAME2023", file_name + ".pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f, encoding='latin1')

v_template = model['v_template']
shapedirs = model['shapedirs']
faces = model['f']
print(f"the keys of the model are {model.keys()}")

# Create random betas as chumpy variable
betas = chumpy.random.randn(400) 

v_shaped = v_template
# v_shaped = v_template + shapedirs.dot(betas)
# print(v_shaped)
def save_obj(path, vertices, faces):
    with open(path, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

save_obj(script_dir + f"/mesh/{file_name}.obj", np.asarray(v_shaped), faces)
print(f"Exported {file_name}.obj")
