import numpy as np

# 加载你的468个关键点 (每个点通常是 [x, y, z])
all_landmarks = np.load("mediapipe_468_landmarks.npy")  # shape: (468, 3)

# 加载embedding文件，获取 105 个索引
embedding = np.load("model/mediapipe_landmark_embedding/mediapipe_landmark_embedding.npz")
indices_105 = embedding['landmark_indices']  # shape: (105,)

landmarks_105 = all_landmarks[indices_105]  # shape: (105, 3)

np.savetxt("2d_mediapipe_105_landmarks.txt", landmarks_105)
