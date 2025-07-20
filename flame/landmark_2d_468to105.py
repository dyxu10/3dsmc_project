import os
import cv2
import numpy as np
import mediapipe as mp

# ==== 设置路径 ====
image_folder = "/Users/chenjueyu/Documents/SS 2025 1.Sem RCI/3DScann/Final Project/3dsmc_project/flame/color"  # 图像文件夹路径
embedding_path = "model/mediapipe_landmark_embedding/mediapipe_landmark_embedding.npz"
output_folder = "/Users/chenjueyu/Documents/SS 2025 1.Sem RCI/3DScann/Final Project/3dsmc_project/flame/2dlandmarks"
os.makedirs(output_folder, exist_ok=True)

# ==== 1. 初始化 MediaPipe Face Mesh ====
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

# ==== 2. 读取 embedding 索引 ====
embedding = np.load(embedding_path)
indices_105 = embedding['landmark_indices']  # shape: (105,)

# ==== 3. 遍历文件夹中的所有 PNG 图像 ====
for filename in os.listdir(image_folder):
    if not filename.lower().endswith(".png"):
        continue

    image_path = os.path.join(image_folder, filename)
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {filename}")
        continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = image.shape
            landmark_array = []
            for lm in face_landmarks.landmark:
                x = lm.x * w
                y = lm.y * h
                landmark_array.append([x, y])
            landmark_array = np.array(landmark_array)  # shape: (468, 2)
            landmarks_105 = landmark_array[indices_105]  # shape: (105, 2)

            # 构造输出路径
            base_name = os.path.splitext(filename)[0]
            output_path = os.path.join(output_folder, base_name + "_2dlandmarks.txt")
            np.savetxt(output_path, landmarks_105, fmt="%.6f")
            print(f"✔ 已保存: {output_path}")
    else:
        print(f"❌ 未检测到人脸: {filename}")