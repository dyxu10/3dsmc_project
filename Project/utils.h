#pragma once
#include <string>
#include <vector>
#include <array>
#include <Eigen/Dense>

using namespace Eigen;

using Vector4f = Eigen::Matrix<float, 4, 1>;
using Vector4uc = Eigen::Matrix<unsigned char, 4, 1>;

#define MINF std::numeric_limits<float>::lowest()

struct Vertex {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Vector4f position;
    Vector4uc color;
};

Eigen::Matrix3f ReadIntrinsics(const std::string& path);
Eigen::Matrix<float, 3, 4> ReadExtrinsics(const std::string& path);
Eigen::Matrix4f ConvertExtrinsicsToHomogeneous(const Eigen::Matrix<float, 3, 4>& ext);
bool WriteMesh(const std::vector<Vertex>& vertices, int width, int height, const std::string& filename);
