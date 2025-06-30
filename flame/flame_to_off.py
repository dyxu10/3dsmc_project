import pickle
import numpy as np
import chumpy
import os

# 这个是把mediapipe_landmark_embedding.npz里的105个landmarks提取出来转换成OFF文件：flame_mediapipe_landmarks.off

def save_landmarks_off(path, landmarks):
    """
    将 landmarks (N×3) 保存为一个 OFF 点云，
    OFF 格式：第一行 “OFF”，第二行 “<num_vertices> 0 0”，
    后面每行一个 “x y z”。
    """
    n = landmarks.shape[0]
    with open(path, 'w') as f:
        f.write("OFF\n")
        f.write(f"{n} 0 0\n")
        for lm in landmarks:
            f.write(f"{lm[0]} {lm[1]} {lm[2]}\n")

# Load FLAME model
script_dir = os.path.dirname(os.path.abspath(__file__))
pkl_path = os.path.join(script_dir, "model/FLAME2023/flame2023.pkl")
with open(pkl_path, "rb") as f:
    model = pickle.load(f, encoding='latin1')

v_template = model['v_template']
faces     = model['f']

# 你也可以启用形状参数：v_shaped = v_template + shapedirs.dot(betas)
v_shaped = v_template

# Load mediapipe landmark embedding
embedding_path = os.path.join(script_dir, "model/mediapipe_landmark_embedding/mediapipe_landmark_embedding.npz")
emb = np.load(embedding_path)
lmk_face_idx = emb['lmk_face_idx']
lmk_b_coords = emb['lmk_b_coords']

# 插值计算 landmarks
landmarks = []
for face_idx, bary in zip(lmk_face_idx, lmk_b_coords):
    tri = faces[face_idx]               # 三角形顶点索引，比如 [i0, i1, i2]
    verts = v_shaped[tri]               # 对应三个顶点的坐标，shape=(3,3)
    lm_pos = np.sum(verts * bary[:,None], axis=0)
    landmarks.append(lm_pos)
landmarks = np.vstack(landmarks)       # (num_landmarks, 3)

# 保存为 OFF
off_path = os.path.join(script_dir, "flame_mediapipe_landmarks.off")
save_landmarks_off(off_path, landmarks)
print(f"Saved {landmarks.shape[0]} landmarks to {off_path}")
