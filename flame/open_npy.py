import numpy as np
import os
import cv2

def convert_npy_to_txt(npy_file_path, txt_file_path=None):
    """
    Convert a .npy file containing 2D facial landmarks to a .txt file
    
    Args:
        npy_file_path (str): Path to the input .npy file
        txt_file_path (str): Path to the output .txt file (optional)
    """

    # Load the numpy array
    landmarks = np.load(npy_file_path)
    print(type(landmarks))
    
    # Generate output filename if not provided
    if txt_file_path is None:
        base_name = os.path.splitext(npy_file_path)[0]
        txt_file_path = base_name + '.txt'
    
    # Print array info
    print(f"Loaded array shape: {landmarks.shape}")
    print(f"Array dtype: {landmarks.dtype}")
    
    # Save as text file
    # Using space as delimiter and preserving precision for float values
    np.savetxt(txt_file_path, landmarks, fmt='%.6f', delimiter=' ')
    
    print(f"Successfully converted {npy_file_path} to {txt_file_path}")
        

def visualize_landmarks(image_path, landmark_txt_path):
    """
    Show the image with landmarks marked as red dots

    Args:
        image_path (str): Path to the image
        landmark_txt_path (str): Path to the landmark .txt file
    """
    image = cv2.imread(image_path)
 

    landmarks = np.loadtxt(landmark_txt_path)
    print(f"number of landmarks:{landmarks.shape[0]}")
    
    if landmarks.ndim == 1:
        landmarks = np.expand_dims(landmarks, axis=0)

    for (x, y) in landmarks:
        cv2.circle(image, (int(round(x)), int(round(y))), radius=2, color=(0, 0, 255), thickness=-1)

    cv2.imshow('Landmarks', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


script_dir = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    landmark_npy_path = script_dir + "/data/color/PIPnet_landmarks/00001.npy"  
    landmark_txt_path = script_dir + "/out/00001.txt"
    image_path = script_dir + "/data/color/00001.png"

    convert_npy_to_txt(landmark_npy_path, landmark_txt_path)
    visualize_landmarks(image_path, landmark_txt_path)

