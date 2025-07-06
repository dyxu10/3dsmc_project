import pickle
import numpy as np
import chumpy

def save_landmarks(path, landmarks):
    with open(path, 'w') as f:
        for i, lm in enumerate(landmarks):
            f.write(f"{i+1} {lm[0]} {lm[1]} {lm[2]}\n")

# Load FLAME model
pkl_path = "./model/FLAME2023/flame2023.pkl"
with open(pkl_path, "rb") as f:
    model = pickle.load(f, encoding='latin1')

v_template = model['v_template']
shapedirs = model['shapedirs']
faces = model['f']

# Generate random shape parameters
betas = chumpy.random.randn(400)
v_shaped = v_template# + shapedirs.dot(betas)

# Load mediapipe landmark embedding
embedding_path = "./model/mediapipe_landmark_embedding/mediapipe_landmark_embedding.npz"
embedding = np.load(embedding_path)
print(type(embedding))
print(embedding.keys())

lmk_face_idx = embedding['lmk_face_idx']      # shape: (num_landmarks,)
lmk_b_coords = embedding['lmk_b_coords']      # shape: (num_landmarks, 3) barycentric weights
# landmark_indices = embedding['landmark_indices']  # optional

landmarks = []
for face_idx, bary in zip(lmk_face_idx, lmk_b_coords):
    tri_vertices_idx = faces[face_idx]   # 3 vertex indices of the triangle face
    tri_vertices = v_shaped[tri_vertices_idx]  # get the 3D vertices for this triangle
    lm_pos = np.sum(tri_vertices * bary[:, None], axis=0)  # weighted sum by barycentric coords
    landmarks.append(lm_pos)

landmarks = np.array(landmarks)

# Save landmarks to file
save_landmarks("flame_mediapipe_landmarks.txt", landmarks)
print(f"Saved {len(landmarks)} landmarks to flame_mediapipe_landmarks.txt")
