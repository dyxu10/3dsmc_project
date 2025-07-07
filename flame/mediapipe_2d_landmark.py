import cv2
import numpy as np
import mediapipe as mp

# 1. 初始化 MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

# 2. 读取图像
image_path = "00001.png"  # 替换为你的图像路径
image = cv2.imread(image_path)
if image is None:
    raise ValueError("图像路径错误或图像无法读取")

# 3. 转换颜色通道
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 4. 获取面部关键点
results = face_mesh.process(image_rgb)

# 5. 提取并保存
if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        landmark_array = []
        h, w, _ = image.shape
        for lm in face_landmarks.landmark:
            x = lm.x * w
            y = lm.y * h
            landmark_array.append([x, y])
        landmark_array = np.array(landmark_array)  # shape (468, 2)

        # 保存为 npy
        np.save("mediapipe_468_landmarks.npy", landmark_array)
        print("保存成功为 mediapipe_468_landmarks.npy")
else:
    print("未检测到人脸")