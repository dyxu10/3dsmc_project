import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

# Load mediapipe landmark embedding

embedding_path = script_dir + "/model/mediapipe_landmark_embedding/00001.npy"
embedding = np.load(embedding_path)
print(type(embedding))
print(embedding.shape)