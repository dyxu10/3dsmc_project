import numpy as np

# Load mediapipe landmark embedding
embedding_path = "./model/mediapipe_landmark_embedding/00001.npy"
embedding = np.load(embedding_path)
print(type(embedding))
print(embedding.shape)