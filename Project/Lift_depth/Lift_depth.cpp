#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

using namespace Eigen;
using namespace cv;

using Vector4f = Eigen::Matrix<float, 4, 1>;
using Vector4uc = Eigen::Matrix<unsigned char, 4, 1>;

#define MINF std::numeric_limits<float>::lowest()

struct Vertex {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Vector4f position;
    Vector4uc color;
};

// Read 3x3 or 4x4 matrix from .txt file
template<int Rows, int Cols>
Matrix<float, Rows, Cols> LoadMatrix(const std::string& path) {
    Matrix<float, Rows, Cols> mat;
    std::ifstream file(path);
    for (int i = 0; i < Rows; ++i)
        for (int j = 0; j < Cols; ++j)
            file >> mat(i, j);
    return mat;
}

bool WriteMesh(const std::vector<Vertex>& vertices, int width, int height, const std::string& filename) {
    std::ofstream out(filename);
    if (!out.is_open()) return false;

    int numVerts = width * height;
    int numFaces = 0;
    std::vector<std::array<int, 3>> faces;

    auto isValid = [&](int i) {
        return vertices[i].position[0] != MINF;
    };

    auto edgeLenSqr = [&](int i1, int i2) {
        return (vertices[i1].position.head<3>() - vertices[i2].position.head<3>()).squaredNorm();
    };

    float threshold = 0.01f;
    float thresholdSqr = threshold * threshold;


    for (int y = 0; y < height - 1; ++y) {
        for (int x = 0; x < width - 1; ++x) {
            int i0 = y * width + x;
            int i1 = y * width + (x + 1);
            int i2 = (y + 1) * width + x;
            int i3 = (y + 1) * width + (x + 1);

            if (isValid(i0) && isValid(i2) && isValid(i1) &&
                edgeLenSqr(i0, i2) < thresholdSqr &&
                edgeLenSqr(i2, i1) < thresholdSqr &&
                edgeLenSqr(i1, i0) < thresholdSqr)
                faces.push_back({i0, i2, i1});

            if (isValid(i1) && isValid(i2) && isValid(i3) &&
                edgeLenSqr(i1, i2) < thresholdSqr &&
                edgeLenSqr(i2, i3) < thresholdSqr &&
                edgeLenSqr(i3, i1) < thresholdSqr)
                faces.push_back({i1, i2, i3});
        }
    }

    out << "COFF\n" << numVerts << " " << faces.size() << " 0\n";

    for (const auto& v : vertices) {
        if (v.position[0] == MINF)
            out << "0 0 0 0 0 0 0\n";
        else
            out << v.position[0] << " " << v.position[1] << " " << v.position[2] << " "
                << static_cast<int>(v.color[0]) << " "
                << static_cast<int>(v.color[1]) << " "
                << static_cast<int>(v.color[2]) << " "
                << static_cast<int>(v.color[3]) << "\n";
    }

    for (auto& f : faces)
        out << "3 " << f[0] << " " << f[1] << " " << f[2] << "\n";

    return true;
}

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
        translation(i) = vals[9 + i] / 1000.0f;  // Convert from mm to meters
    }

    Eigen::Matrix<float, 3, 4> extrinsic;
    extrinsic.block<3, 3>(0, 0) = rotation;
    extrinsic.col(3) = translation;

    return extrinsic;
}


Eigen::Matrix4f ConvertExtrinsicsToHomogeneous(const Eigen::Matrix<float, 3, 4>& ext3x4) {
    Eigen::Matrix4f ext4x4 = Eigen::Matrix4f::Identity();
    ext4x4.block<3,4>(0,0) = ext3x4;
    return ext4x4;
}


Eigen::Matrix3f ReadIntrinsics(const std::string& path) {
    std::ifstream file(path);
    std::vector<float> vals;
    float val;
    
    while (file >> val) {
        vals.push_back(val);
    }

    if (vals.size() < 4) {
        throw std::runtime_error("Intrinsic file must contain at least 4 floats: fx, fy, cx, cy");
    }

    float fx = vals[0];
    float fy = vals[1];
    float cx = vals[2];
    float cy = vals[3];

    Eigen::Matrix3f intrinsic = Eigen::Matrix3f::Identity();
    intrinsic(0, 0) = fx;
    intrinsic(1, 1) = fy;
    intrinsic(0, 2) = cx;
    intrinsic(1, 2) = cy;

    return intrinsic;
}

int main() {

    std::string frame = "00001";
    std::string colorPath = "../data/color/" + frame + ".png";
    std::string depthPath = "../data/depth/" + frame + ".png";

    std::string depthIntrPath = "../data/camera/c00_depth_intrinsic.txt";
    std::string colorIntrPath = "../data/camera/c00_color_intrinsic.txt";

    Matrix3f K_depth = ReadIntrinsics(depthIntrPath);
    Matrix3f K_color = ReadIntrinsics(colorIntrPath);

    std::cout << "K_color:\n" << K_color << std::endl;
    std::cout << "K_depth:\n" << K_depth << std::endl;

    std::string depthExtrPath = "../data/camera/c00_depth_extrinsic.txt";
    std::string colorExtrPath = "../data/camera/c00_color_extrinsic.txt";

    auto E_depth = ReadExtrinsics(depthExtrPath);
    auto E_color = ReadExtrinsics(colorExtrPath);

    auto E_depth_4x4 = ConvertExtrinsicsToHomogeneous(E_depth);
    auto E_color_4x4 = ConvertExtrinsicsToHomogeneous(E_color);

    std::cout << "E_color:\n" << E_color << std::endl;
    std::cout << "E_depth:\n" << E_depth << std::endl;

    auto color = imread(colorPath, IMREAD_UNCHANGED); 
    auto depth = imread(depthPath, IMREAD_UNCHANGED); 

    std::cout << "color image size: " << color.size() << std::endl;
    std::cout << "depth image size: " << depth.size() << std::endl;


    
    // std::cout << "color image type:\n" << typeid(color).name() << std::endl;
    // std::cout << "depth image type:\n" << typeid(depth).name() << std::endl;
    // std::cout << "depth type: " << depth.type() << " (CV_16U is " << CV_16U << ", CV_32F is " << CV_32F << ")" << std::endl;

    if (depth.empty() || color.empty()) {
        std::cerr << "Failed to load images.\n";
        return -1;
    }

    int width = depth.cols;
    int height = depth.rows;
    std::vector<Vertex> vertices(width * height);

    // DPHM-style approach: work directly in depth camera coordinates
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            ushort d_raw = depth.at<ushort>(y, x);
            float d = d_raw / 1000.0f; //convert to meters

            if (d_raw >= 700 || d_raw < 0.1f) {
                vertices[idx].position = Vector4f(MINF, MINF, MINF, MINF);
                vertices[idx].color = Vector4uc(0, 0, 0, 0);
                continue;
            }


            float X = (x - K_color(0,2)) * d / K_color(0,0);
            float Y = (y - K_color(1,2)) * d / K_color(1,1);
            float Z = d;

            // Extrinsic : world -> camera
            // Extrinsic inverse : camera -> world
            // Intrinsic : camera -> pixel
            // Intrinsic inverse : pixel -> camera
            // Vector4f P_depth(X, Y, Z, 1.0f);
            // Vector4f P_world = E_depth_4x4.inverse() * P_depth; 
            // Vector4f P_color = E_color_4x4 * P_world;

            Vector4f P_color(X, Y, Z, 1.0f);

            vertices[idx].position = P_color;
            
            Vec3b rgb = color.at<Vec3b>(y, x);
            vertices[idx].color = Vector4uc(rgb[2], rgb[1], rgb[0], 255);
        }
    }

    WriteMesh(vertices, width, height, "../out/output.off");
    std::cout << "Mesh written to output.off" << std::endl;
    return 0;
}