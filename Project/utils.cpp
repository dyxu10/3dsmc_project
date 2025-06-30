#include "utils.h"
#include <fstream>
#include <limits>
#include <stdexcept>
#include <vector>
#include <array>
#include <iterator>

using namespace Eigen;

Matrix<float, 3, 4> ReadExtrinsics(const std::string& path) {
    std::ifstream file(path);
    std::vector<float> vals((std::istream_iterator<float>(file)), std::istream_iterator<float>());
    if (vals.size() != 12) throw std::runtime_error("Extrinsic must have 12 values");
    Matrix<float, 3, 4> ext;
    for (int i = 0; i < 12; ++i) ext(i / 4, i % 4) = vals[i];
    return ext;
}

Matrix3f ReadIntrinsics(const std::string& path) {
    std::ifstream file(path);
    std::vector<float> vals((std::istream_iterator<float>(file)), std::istream_iterator<float>());
    if (vals.size() < 4) throw std::runtime_error("Intrinsic must have at least 4 values");
    Matrix3f K = Matrix3f::Identity();
    K(0, 0) = vals[0];
    K(1, 1) = vals[1];
    K(0, 2) = vals[2];
    K(1, 2) = vals[3];
    return K;
}

Matrix4f ConvertExtrinsicsToHomogeneous(const Matrix<float, 3, 4>& ext3x4) {
    Matrix4f ext4x4 = Matrix4f::Identity();
    ext4x4.block<3, 4>(0, 0) = ext3x4;
    return ext4x4;
}

bool WriteMesh(const std::vector<Vertex>& vertices, int width, int height, const std::string& filename) {
    std::ofstream out(filename);
    if (!out.is_open()) return false;

    const float threshold = 0.01f;
    const float thresholdSqr = threshold * threshold;

    auto isValid = [&](int i) {
        return vertices[i].position[0] != MINF;
    };
    auto edgeLenSqr = [&](int i1, int i2) {
        return (vertices[i1].position.head<3>() - vertices[i2].position.head<3>()).squaredNorm();
    };

    std::vector<std::array<int, 3>> faces;
    for (int y = 0; y < height - 1; ++y) {
        for (int x = 0; x < width - 1; ++x) {
            int i0 = y * width + x;
            int i1 = i0 + 1;
            int i2 = i0 + width;
            int i3 = i2 + 1;

            if (isValid(i0) && isValid(i1) && isValid(i2) &&
                edgeLenSqr(i0, i1) < thresholdSqr &&
                edgeLenSqr(i1, i2) < thresholdSqr &&
                edgeLenSqr(i2, i0) < thresholdSqr) {
                faces.push_back({i0, i1, i2});
            }
            if (isValid(i1) && isValid(i2) && isValid(i3) &&
                edgeLenSqr(i1, i2) < thresholdSqr &&
                edgeLenSqr(i2, i3) < thresholdSqr &&
                edgeLenSqr(i3, i1) < thresholdSqr) {
                faces.push_back({i1, i2, i3});
            }
        }
    }

    out << "COFF\n" << (width * height) << " " << faces.size() << " 0\n";

    for (const auto& v : vertices) {
        if (v.position[0] == MINF) {
            out << "0 0 0 0 0 0 0\n";
        } else {
            out << v.position[0] << " "
                << v.position[1] << " "
                << v.position[2] << " "
                << static_cast<int>(v.color[0]) << " "
                << static_cast<int>(v.color[1]) << " "
                << static_cast<int>(v.color[2]) << " "
                << static_cast<int>(v.color[3]) << "\n";
        }
    }

    for (const auto& f : faces) {
        out << "3 " << f[0] << " " << f[1] << " " << f[2] << "\n";
    }

    return true;
}
