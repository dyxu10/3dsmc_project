// LossTest.cpp
// 用mean的flame脸提取出来的所有点flame2023_mean.txt当作目标点，
// 直接从flame模型里检索出来的点当作sourcePoints，flame模型的betas当作优化目标。
// 如果lossfuncton没有问题的话那我的betas优化结果应该全部是0。

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>               // for std::memcpy

#include <Eigen/Dense>
#include <ceres/ceres.h>
#include "cnpy.h"

// —— 全局变量 ——
static Eigen::MatrixXd templateVertices;                // double precision
static std::vector<double>    shapeDirections;          // double precision
static std::vector<Eigen::Vector3i> faces;
static int numVertices        = 0;
static int numShapeParameters = 0;
static int numFaces           = 0;

// 读取 numVertices × 3 的目标点 (x y z 每行)，转换成3 × numVertices的矩阵
static Eigen::MatrixXd loadTargetPoints(const std::string &filePath) {
    std::ifstream inputStream(filePath);
    if (!inputStream)
        throw std::runtime_error("Cannot open target points file: " + filePath);

    Eigen::MatrixXd targetMatrix(3, numVertices);
    for (int i = 0; i < numVertices; ++i) {
        double x, y, z;
        if (!(inputStream >> x >> y >> z))
            throw std::runtime_error("Malformed or too few lines in target points file");
        targetMatrix(0, i) = x;
        targetMatrix(1, i) = y;
        targetMatrix(2, i) = z;
    }
    return targetMatrix;
}

// 点到点残差
struct P2PointResidual {
    P2PointResidual(int vertexIndex, const Eigen::Vector3d &targetPoint)
      : vertexIndex_(vertexIndex), targetPoint_(targetPoint) {}

    template <typename T>
    bool operator()(T const* const* parameters, T* residuals) const {
        const T* shapeParams = parameters[0];

        // 模板顶点
        T px = T(templateVertices(vertexIndex_, 0));
        T py = T(templateVertices(vertexIndex_, 1));
        T pz = T(templateVertices(vertexIndex_, 2));

        // 叠加形变
        for (int k = 0; k < numShapeParameters; ++k) {
            T w = shapeParams[k];
            px += T(shapeDirections[(vertexIndex_*3 + 0)*numShapeParameters + k]) * w;
            py += T(shapeDirections[(vertexIndex_*3 + 1)*numShapeParameters + k]) * w;
            pz += T(shapeDirections[(vertexIndex_*3 + 2)*numShapeParameters + k]) * w;
        }

        residuals[0] = px - T(targetPoint_(0));
        residuals[1] = py - T(targetPoint_(1));
        residuals[2] = pz - T(targetPoint_(2));
        return true;
    }

    int vertexIndex_;
    Eigen::Vector3d targetPoint_;
};

void saveVerticesAsTxt(const Eigen::MatrixXd& vertices, const std::string& path) {
    std::ofstream out(path);
    if (!out.is_open()) {
        std::cerr << "Could not open file to write vertices: " << path << std::endl;
        return;
    }

    for (int i = 0; i < vertices.rows(); ++i) {
        out << vertices(i, 0) << " "
            << vertices(i, 1) << " "
            << vertices(i, 2) << "\n";
    }
    out.close();
    std::cout << "Saved mean FLAME face to " << path << std::endl;
}

int main() {
    const std::string flameModel  = "../model/FLAME2023/flame2023_no_jaw.npz";
    const std::string targetsFile = "../Data/3d/flame2023_mean_test.txt";


    // 1. 加载 FLAME 模型
    auto vTpl   = cnpy::npz_load(flameModel, "v_template");
    auto sDirs  = cnpy::npz_load(flameModel, "shapedirs");
    auto fArr   = cnpy::npz_load(flameModel, "faces");

    numVertices        = int(vTpl.shape[0]);
    numShapeParameters = int(sDirs.shape[2]);
    // numFaces           = int(fArr.shape[0]);

    // 模板顶点（double）
    templateVertices.resize(numVertices, 3);
    std::memcpy(
        templateVertices.data(),
        vTpl.data<double>(),
        sizeof(double) * numVertices * 3
    );

    // 形变方向
    shapeDirections.resize(numVertices * 3 * numShapeParameters);
    std::memcpy(
        shapeDirections.data(),
        sDirs.data<double>(),
        sizeof(double) * shapeDirections.size()
    );

    // faces
    // int* f_data = fArr.data<int>(); 
    // faces.resize(numFaces);
    // for (int i = 0; i < numFaces; ++i) {
    //     faces[i] = Eigen::Vector3i(
    //         f_data[3*i+0],
    //         f_data[3*i+1],
    //         f_data[3*i+2]
    //     );
    // }

    // saveVerticesAsTxt(templateVertices, "../Data/3d/flame2023_mean_test.txt");
    // const std::string targetsFile = "../Data/3d/flame2023_mean_test.txt";


    // 2. 加载目标点 (3 × numVertices)
    Eigen::MatrixXd targets = loadTargetPoints(targetsFile);

    // 3. 初始化参数
    std::vector<double> shapeParameters(numShapeParameters, 0.0);

    // 5. 构造 Ceres 问题
    ceres::Problem problem;
    problem.AddParameterBlock(shapeParameters.data(), numShapeParameters);

    for (int vi = 0; vi < numVertices; ++vi) {
        // P2P
        auto* cost_p2p = new ceres::DynamicAutoDiffCostFunction<P2PointResidual>(
            new P2PointResidual(vi, targets.col(vi).eval())
        );
        cost_p2p->AddParameterBlock(numShapeParameters);
        cost_p2p->SetNumResiduals(3);
        problem.AddResidualBlock(cost_p2p, nullptr, shapeParameters.data());
    }

    // 6. 求解
    ceres::Solver::Options opts;
    opts.trust_region_strategy_type  = ceres::LEVENBERG_MARQUARDT;
    opts.use_nonmonotonic_steps       = false;
    opts.linear_solver_type           = ceres::DENSE_QR;
    opts.minimizer_progress_to_stdout = 1;
    opts.num_threads                  = 8;

    ceres::Solver::Summary summary;
    ceres::Solve(opts, &problem, &summary);
    std::cout << summary.FullReport() << std::endl;

    // 7. 打印最终 loss
    double totalLoss = 0.0;
    for (int i = 0; i < numVertices; ++i) {
        Eigen::Vector3d v = templateVertices.row(i).transpose();
        for (int k = 0; k < numShapeParameters; ++k) {
            v[0] += shapeDirections[(i * 3 + 0)*numShapeParameters + k] * shapeParameters[k];
            v[1] += shapeDirections[(i * 3 + 1)*numShapeParameters + k] * shapeParameters[k];
            v[2] += shapeDirections[(i * 3 + 2)*numShapeParameters + k] * shapeParameters[k];
        }
        Eigen::Vector3d diff = v - targets.col(i).eval();
        totalLoss += diff.squaredNorm();
    }
    std::cout << "Final loss (Ceres style, cost): " << totalLoss / 2.0 << std::endl;

    // 8. 保存 betas
    std::ofstream betaFile("../Data/3d/test_betas_3.txt");
    for (double b : shapeParameters) betaFile << b << "\n";
    betaFile.close();
    std::cout << "Saved shape parameters to test_betas_3.txt\n";

    return 0;
}
