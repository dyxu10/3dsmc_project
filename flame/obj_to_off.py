import numpy as np

def save_npy_as_off(npy_path, off_path):
    # 读取 npy 文件
    points = np.load(npy_path)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Expected shape (N, 3) for 3D point cloud")

    num_points = points.shape[0]

    with open(off_path, 'w') as f:
        f.write("OFF\n")
        f.write(f"{num_points} 0 0\n")
        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")

    print(f"Saved OFF file: {off_path}")

# 使用方法
save_npy_as_off("00001_point3d.npy", "00001_point3d.off")