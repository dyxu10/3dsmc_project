// compute_loss.cpp

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include <Eigen/Dense>
#include <ceres/ceres.h>
#include "cnpy.h"

// —— 全局变量 ——
// 模板顶点 (numVertices × 3)
static Eigen::MatrixXd templateVertices;
// 形状方向 (numVertices * 3 * numShapeParameters)
static std::vector<double> shapeDirections;
static int numVertices = 0;
static int numShapeParameters = 0;

// 读取 numVertices × 3 的目标点 (x y z 每行)
static Eigen::MatrixXd loadTargetPoints(const std::string &filePath) {
    std::ifstream inputStream(filePath);
    if (!inputStream) throw std::runtime_error("Cannot open target points file: " + filePath);
    Eigen::MatrixXd targetMatrix(3, numVertices);
    for (int i = 0; i < numVertices; ++i) {
        double x, y, z;
        if (!(inputStream >> x >> y >> z)) {
            throw std::runtime_error("Malformed or too few lines in target points file");
        }
        targetMatrix(0, i) = x;
        targetMatrix(1, i) = y;
        targetMatrix(2, i) = z;
    }
    return targetMatrix;
}

// 动态自动微分残差结构体
struct ShapeParameterResidual {
    ShapeParameterResidual(int vertexIndex, const Eigen::Vector3d &targetPoint)
        : vertexIndex(vertexIndex), targetPoint(targetPoint) {}

    template <typename T>
    bool operator()(T const* const* parameters, T* residuals) const {
        // parameters[0] 是形状参数数组
        const T* shapeParams = parameters[0];
        // 获取模板顶点
        T px = T(templateVertices(vertexIndex, 0));
        T py = T(templateVertices(vertexIndex, 1));
        T pz = T(templateVertices(vertexIndex, 2));
        // 叠加形变
        for (int k = 0; k < numShapeParameters; ++k) {
            T w = shapeParams[k];
            px += T(shapeDirections[(vertexIndex*3 + 0) * numShapeParameters + k]) * w;
            py += T(shapeDirections[(vertexIndex*3 + 1) * numShapeParameters + k]) * w;
            pz += T(shapeDirections[(vertexIndex*3 + 2) * numShapeParameters + k]) * w;
        }
        // 计算残差 = p - target
        residuals[0] = px - T(targetPoint(0));
        residuals[1] = py - T(targetPoint(1));
        residuals[2] = pz - T(targetPoint(2));
        return true;
    }

    int vertexIndex;
    Eigen::Vector3d targetPoint;
};

int main() {
    const std::string flameModel = "../model/FLAME2023/flame2023_no_jaw.npz";
    const std::string targetsFile = "../Data/3d/flame2output_points_xyz.txt";

    // 1. 加载 FLAME 模型
    auto vTemplate = cnpy::npz_load(flameModel, "v_template");
    auto shapedirs = cnpy::npz_load(flameModel, "shapedirs");
    numVertices = int(vTemplate.shape[0]);
    numShapeParameters = int(shapedirs.shape[2]);

    templateVertices.resize(numVertices, 3);
    std::memcpy(templateVertices.data(), vTemplate.data<double>(), sizeof(double) * numVertices * 3);

    shapeDirections.resize(numVertices * 3 * numShapeParameters);
    std::memcpy(shapeDirections.data(), shapedirs.data<double>(), sizeof(double) * shapeDirections.size());

    // 2. 加载目标点，并转换为列向量
    Eigen::MatrixXd targets = loadTargetPoints(targetsFile);

    // 3. 初始化参数
    std::vector<double> shapeParameters(numShapeParameters, 0.0);

    // 4. 构造 Ceres 问题
    ceres::Problem problem;
    problem.AddParameterBlock(shapeParameters.data(), numShapeParameters);
    for (int i = 0; i < numVertices; ++i) {
        auto* cost = new ceres::DynamicAutoDiffCostFunction<ShapeParameterResidual>(
            new ShapeParameterResidual(i, targets.col(i))
        );
        cost->AddParameterBlock(numShapeParameters);
        cost->SetNumResiduals(3);
        problem.AddResidualBlock(cost, nullptr, shapeParameters.data());
    }

    // 5. 求解
    ceres::Solver::Options opts;
    opts.linear_solver_type = ceres::DENSE_QR;
    opts.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(opts, &problem, &summary);
    std::cout << summary.FullReport() << std::endl;

    // ---------- 打印最终 loss ----------
    double totalLoss = 0.0;
    for (int i = 0; i < numVertices; ++i) {
        Eigen::Vector3d v = templateVertices.row(i).transpose();

        for (int k = 0; k < numShapeParameters; ++k) {
            v[0] += shapeDirections[(i * 3 + 0) * numShapeParameters + k] * shapeParameters[k];
            v[1] += shapeDirections[(i * 3 + 1) * numShapeParameters + k] * shapeParameters[k];
            v[2] += shapeDirections[(i * 3 + 2) * numShapeParameters + k] * shapeParameters[k];
        }
        Eigen::Vector3d diff = v - targets.col(i);

        totalLoss += diff.squaredNorm();
    }
    std::cout << "Final loss (Ceres style, cost): " << totalLoss / 2.0 << std::endl;


    // ---------- 保存 betas ----------
    std::ofstream betaFile("optimized_betas.txt");
    for (double b : shapeParameters) betaFile << b << "\n";
    betaFile.close();
    std::cout << "Saved shape parameters to optimized_betas.txt" << std::endl;
    

    return 0;
}
