// FLAME 3D Point Cloud Transformation Pipeline
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

using namespace Eigen;
using namespace cv;

struct Vertex {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Vector3f position;
    Vector4i color;
};

Matrix3f ReadIntrinsics(const std::string& path) {
    std::ifstream file(path);
    std::vector<float> vals;
    float val;
    while (file >> val) vals.push_back(val);
    if (vals.size() < 4) throw std::runtime_error("Invalid intrinsics");
    Matrix3f K = Matrix3f::Identity();
    K(0,0) = vals[2]; K(1,1) = vals[3];
    K(0,2) = vals[0]; K(1,2) = vals[1];
    return K;
}

std::vector<Vector2f> LoadLandmarks2D(const std::string& path) {
    std::vector<Vector2f> points;
    std::ifstream file(path);
    float x, y;
    while (file >> x >> y) points.emplace_back(x, y);
    return points;
}

MatrixXf LoadLandmarks3D(const std::string& filename) {
    std::ifstream file(filename);
    std::vector<Vector3f> pts;
    float x, y, z;
    while (file >> x >> y >> z) pts.emplace_back(x, y, z);
    MatrixXf mat(pts.size(), 3);
    for (size_t i = 0; i < pts.size(); ++i) mat.row(i) = pts[i];
    return mat;
}

float ComputeRMSNorm(const MatrixXf& pts) {
    MatrixXf centered = pts.rowwise() - pts.colwise().mean();
    return std::sqrt((centered.array().square().rowwise().sum().mean()));
}

void RigidAlignment(const MatrixXd& src, const MatrixXd& tgt, Matrix3d& R, Vector3d& T) {
    Vector3d src_mean = src.colwise().mean(), tgt_mean = tgt.colwise().mean();
    MatrixXd src_c = src.rowwise() - src_mean.transpose();
    MatrixXd tgt_c = tgt.rowwise() - tgt_mean.transpose();
    Matrix3d H = src_c.transpose() * tgt_c;
    JacobiSVD<MatrixXd> svd(H, ComputeFullU | ComputeFullV);
    Matrix3d U = svd.matrixU(), V = svd.matrixV();
    R = V * U.transpose();
    if (R.determinant() < 0) { V.col(2) *= -1; R = V * U.transpose(); }
    T = tgt_mean - R * src_mean;
}

int main() {
    std::string frame = "00001";
    std::string colorPath = "../dataset/color/" + frame + ".png";
    std::string depthPath = "../dataset/depth/" + frame + ".png";
    std::string landmarkPath = "../dataset/2dlandmarks/" + frame + "_2dlandmarks.txt";

    Matrix3f K = ReadIntrinsics("../dataset/camera/c00_color_intrinsic.txt");
    MatrixXf flame_lms = LoadLandmarks3D("../dataset/flame_mediapipe_landmarks.txt");

    Mat color = imread(colorPath, IMREAD_UNCHANGED);
    Mat depth = imread(depthPath, IMREAD_UNCHANGED);
    if (color.empty() || depth.empty()) throw std::runtime_error("Cannot load images");

   //Landmark3D
    std::vector<Vector2f> landmarks2D = LoadLandmarks2D(landmarkPath);
    std::vector<Vector3f> landmarks3D;
    for (const auto& pt : landmarks2D) {
        int x = int(pt.x()), y = int(pt.y());
        if (x < 0 || x >= depth.cols || y < 0 || y >= depth.rows) continue;
        ushort d_raw = depth.at<ushort>(y, x);
        if (d_raw == 0 || d_raw >= 700) continue;
        float d = d_raw / 1000.0f;
        float X = (x - K(0,2)) * d / K(0,0);
        float Y = (y - K(1,2)) * d / K(1,1);
        landmarks3D.emplace_back(X, Y, d);
    }

    //Scale , R, T
    MatrixXf your_lms(landmarks3D.size(), 3);
    for (size_t i = 0; i < landmarks3D.size(); ++i) your_lms.row(i) = landmarks3D[i];
    float scale = ComputeRMSNorm(flame_lms) / ComputeRMSNorm(your_lms);
    MatrixXd source = your_lms.cast<double>() * scale;
    MatrixXd target = flame_lms.cast<double>();
    Matrix3d R;
    Vector3d T;
    RigidAlignment(source, target, R, T);

    //3d points *RT
    std::vector<Vertex> cloud;
    for (int y = 0; y < depth.rows; ++y) {
        for (int x = 0; x < depth.cols; ++x) {
            ushort d_raw = depth.at<ushort>(y, x);
            if (d_raw == 0 || d_raw >= 700) continue;
            float d = d_raw / 1000.0f;

            float X = (x - K(0,2)) * d / K(0,0);
            float Y = (y - K(1,2)) * d / K(1,1);
            Vector3f p_cam(X, Y, d);
            p_cam *= scale;

            Vector3d p = p_cam.cast<double>();
            p = R * p + T;
            Vertex v;
            v.position = p.cast<float>();

            Vec3b rgb = color.at<Vec3b>(y, x);
            v.color = Vector4i(rgb[2], rgb[1], rgb[0], 255);
            cloud.push_back(v);
        }
    }

    //save
    std::string filename = "../model/mesh/" + frame + "/transformed_" + frame + ".off";
    std::ofstream meshOut(filename);
    if (!meshOut.is_open()) {
        std::cerr << "无法写入: " << filename << std::endl;
        return 1;
    }

    meshOut << "COFF\n" << cloud.size() << " 0 0\n";
    for (const auto& v : cloud) {
        meshOut << v.position.transpose() << " "
                << v.color[0] << " " << v.color[1] << " "
                << v.color[2] << " " << v.color[3] << "\n";
    }

    std::cout << "Saved transformed point cloud with color: " << filename << "\n";
    return 0;
}