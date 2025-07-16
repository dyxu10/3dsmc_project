//将landmark（depth camera）变化到 rgb camera

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

using Vector4f = Eigen::Matrix<float, 4, 1>;

Eigen::Matrix<float, 3, 4> ReadExtrinsics(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open extrinsic file: " + path);
    }

    std::vector<float> vals;
    float val;
    while (file >> val) {
        vals.push_back(val);
    }

    if (vals.size() != 12) {
        throw std::runtime_error("Extrinsic file must contain exactly 12 floats (9 for rotation, 3 for translation)");
    }

    Eigen::Matrix3f rotation;
    Eigen::Vector3f translation;

    for (int i = 0; i < 9; ++i) {
        rotation(i / 3, i % 3) = vals[i];
    }
    for (int i = 0; i < 3; ++i) {
        translation(i) = vals[9 + i];
    }

    Eigen::Matrix<float, 3, 4> extrinsic;
    extrinsic.block<3, 3>(0, 0) = rotation;
    extrinsic.col(3) = translation;

    return extrinsic;
}

// 将3x4扩展为4x4齐次矩阵
Matrix4f ConvertToHomogeneous(const Matrix<float, 3, 4>& ext3x4) {
    Matrix4f ext4x4 = Matrix4f::Identity();
    ext4x4.block<3, 4>(0, 0) = ext3x4;
    return ext4x4;
}

// 读取点（每行x y z）
std::vector<Vector3f> ReadPointsFromTXT(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) throw std::runtime_error("Cannot open: " + filename);

    std::vector<Vector3f> points;
    float x, y, z;
    while (file >> x >> y >> z) {
        points.emplace_back(x, y, z);
    }

    return points;
}

// 保存点
void SavePointsToTXT(const std::string& filename, const std::vector<Vector3f>& points) {
    std::ofstream file(filename);
    if (!file.is_open()) throw std::runtime_error("Cannot write: " + filename);

    for (const auto& p : points) {
        file << p[0] << " " << p[1] << " " << p[2] << "\n";
    }
}

Eigen::Matrix4f ConvertExtrinsicsToHomogeneous(const Eigen::Matrix<float, 3, 4>& ext3x4) {
    Eigen::Matrix4f ext4x4 = Eigen::Matrix4f::Identity();
    ext4x4.block<3,4>(0,0) = ext3x4;
    return ext4x4;
}

int main() {
    try {
        // 路径
        std::string depthExtrPath = "../data/camera/c00_depth_extrinsic.txt";
        std::string colorExtrPath = "../data/camera/c00_color_extrinsic.txt";
        std::string inputPath = "../out/landmarks3D.txt";
        std::string outputPath = "../data/landmarks3D.txt";

        // 读取并转为4x4

        Matrix<float, 3, 4> E_color = ReadExtrinsics(colorExtrPath);
        Matrix<float, 3, 4> E_depth = ReadExtrinsics(depthExtrPath);

        Matrix4f E_color_4x4 = ConvertExtrinsicsToHomogeneous(E_color);
        Matrix4f E_depth_4x4 = ConvertExtrinsicsToHomogeneous(E_depth);


        std::cout << "E_depth:\n" << E_depth << std::endl;
        std::cout << "E_color:\n" << E_color << std::endl;

        // 从 depth 相机坐标系 → RGB 相机坐标系
        Matrix4f E = E_color_4x4 * E_depth_4x4.inverse();
        std::cout << "Transform matrix E (depth → rgb):\n" << E << std::endl;

        // 读取 pip_matching 点
        auto points_depth = ReadPointsFromTXT(inputPath);

        // 转换点
        std::vector<Vector3f> points_rgb;
        for (const auto& p : points_depth) {
            Vector4f p_homo(p[0], p[1], p[2], 1.0f);
            Vector4f p_rgb = E * p_homo;
            points_rgb.emplace_back(p_rgb.head<3>());
        }

        // 写入结果
        SavePointsToTXT(outputPath, points_rgb);
        std::cout << "Transformed points written to: " << outputPath << "\n";

    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return -1;
    }

    return 0;
}