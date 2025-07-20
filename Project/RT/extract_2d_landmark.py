import cv2
import numpy as np
import mediapipe as mp

# ==== 设置路径 ====
image_path = "../dataset/color/00001.png"  # 图像路径
embedding_path = "../model/mediapipe_landmark_embedding/mediapipe_landmark_embedding.npz"
output_txt = "00001.txt"

# ==== 1. 初始化 MediaPipe Face Mesh ====
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

# ==== 2. 读取图像 ====
image = cv2.imread(image_path)
if image is None:
    raise ValueError(f"图像无法读取: {image_path}")

# ==== 3. 转换颜色通道 ====
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ==== 4. 处理图像，提取关键点 ====
results = face_mesh.process(image_rgb)

if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        h, w, _ = image.shape
        landmark_array = []
        for lm in face_landmarks.landmark:
            x = lm.x * w
            y = lm.y * h
            landmark_array.append([x, y])
        landmark_array = np.array(landmark_array)  # shape (468, 3)

        # ==== 5. 提取 105 个索引 ====
        embedding = np.load(embedding_path)
        indices_105 = embedding['landmark_indices']  # shape: (105,)
        landmarks_105 = landmark_array[indices_105]  # shape: (105, 3)

        # ==== 6. 保存为 .txt ====
        np.savetxt(output_txt, landmarks_105, fmt="%.6f")
        print(f"成功保存 105 个关键点为: {output_txt}")
else:
    print("未检测到人脸")