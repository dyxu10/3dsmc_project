import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def plot_3d_landmarks(csv_path):
    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    # Read CSV file
    df = pd.read_csv(csv_path)
    if not all(col in df.columns for col in ['id', 'x', 'y', 'z']):
        print("Error: CSV must contain 'id', 'x', 'y', 'z' columns.")
        return

    # Extract coordinates
    x = df['x'].values
    y = df['y'].values
    z = df['z'].values
    ids = df['id'].values

    # Create 3D scatter plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot points
    scatter = ax.scatter(x, y, z, c='b', marker='o', s=50)

    # Label key landmarks (e.g., nose tip, chin, eyes)
    key_landmarks = {
        0: 'Chin',        # LANDMARK_IDS_68[0] = 234 (jawline)
        27: 'Nose Tip',   # LANDMARK_IDS_68[27] = 1 (nose tip)
        35: 'Right Eye',  # LANDMARK_IDS_68[35] = 33 (right eye)
        41: 'Left Eye',   # LANDMARK_IDS_68[41] = 362 (left eye)
        48: 'Mouth Left'  # LANDMARK_IDS_68[48] = 61 (mouth corner)
    }
    for idx, label in key_landmarks.items():
        if idx < len(ids):
            ax.text(x[idx], y[idx], z[idx], label, size=10, zorder=1, color='r')

    # Set labels and title
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_zlabel('Z (normalized depth)')
    ax.set_title('3D Facial Landmarks')

    # Invert Y-axis to match image coordinates (origin at top-left)
    ax.invert_yaxis()

    # Show plot
    plt.show()

if __name__ == "__main__":
    csv_path = "output/3D.csv"
    plot_3d_landmarks(csv_path)