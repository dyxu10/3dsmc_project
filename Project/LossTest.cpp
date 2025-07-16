// LossTest.cpp
// 用mean的flame脸提取出来的所有点flame2023_mean.txt当作目标点，
// 直接从flame模型里检索出来的点当作sourcePoints，flame模型的betas当作优化目标。
// 如果lossfuncton没有问题的话那我的betas优化结果应该全部是0。

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <Eigen/Dense>
#include <ceres/ceres.h>
#include "cnpy.h"

// —— 全局变量 ——
static Eigen::MatrixXd templateVertices;           
static std::vector<double>    shapeDirections;          
static std::vector<Eigen::Vector3i> faces;
static int numVertices        = 0;
static int numShapeParameters = 0;
static int numFaces           = 0;
static std::vector<int> indexList;
static const int NUM_MACTHED_POINTS = 3430;
static const int ITERATION = 2; //用来记录这是第几轮优化（loss+knn算一轮）

// 读取 3248 × 3 的目标点 (x y z 每行)，转换成3 × 3248的矩阵
static Eigen::MatrixXd loadTargetPoints(const std::string &filePath) {
    std::ifstream inputStream(filePath);
    if (!inputStream)
        throw std::runtime_error("Cannot open target points file: " + filePath);

    Eigen::MatrixXd targetMatrix(3, NUM_MACTHED_POINTS);
    for (int i = 0; i < NUM_MACTHED_POINTS; ++i) {
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

//用来读取index然后存为indexList的
std::vector<int> loadIndexList(const std::string& filePath) {
    std::ifstream file(filePath);

    if (!file.is_open()) {
        throw std::runtime_error("Failed to open index file: " + filePath);
    }

    int idx;
    while (file >> idx) {
        indexList.push_back(idx);
    }

    file.close();
    return indexList;
}

int main() {
    const std::string flameModel  = "../model/FLAME2023/flame2023_new.npz";
    const std::string targetsFile = "../Data/3d/knn/matched_" + std::to_string(ITERATION) + ".txt";
    //读取用到的flame点的index并存在indexList里
    std::string indexFile = "../Data/3d/knn/indices_" + std::to_string(ITERATION) + ".txt";
    indexList = loadIndexList(indexFile);

    // 1. 加载 FLAME 模型
    auto vTpl   = cnpy::npz_load(flameModel, "v_template");
    auto sDirs  = cnpy::npz_load(flameModel, "shapedirs");
    // auto fArr   = cnpy::npz_load(flameModel, "faces");

    numVertices        = int(vTpl.shape[0]);
    numShapeParameters = int(sDirs.shape[2]);
    // numFaces           = int(fArr.shape[0]);

    // 模板顶点（double）
    templateVertices.resize(numVertices, 3);
    const double* vtpl_data = vTpl.data<double>();

    for (int i = 0; i < numVertices; ++i) {
        templateVertices(i, 0) = vtpl_data[i * 3 + 0];  // x
        templateVertices(i, 1) = vtpl_data[i * 3 + 1];  // y
        templateVertices(i, 2) = vtpl_data[i * 3 + 2];  // z
    }


    // 形变方向
    shapeDirections.resize(numVertices * 3 * numShapeParameters);
    const double* sdir_data = sDirs.data<double>();
    for (int v = 0; v < numVertices; ++v) {
        for (int c = 0; c < 3; ++c) {  // 0: x, 1: y, 2: z
            for (int b = 0; b < numShapeParameters; ++b) {
                // 3D 到扁平化索引：FLAME 风格展开成 V*3 行 × B 列
                int flatIndex = (v * 3 + c) * numShapeParameters + b;
                int npyIndex = v * 3 * numShapeParameters + c * numShapeParameters + b;

                shapeDirections[flatIndex] = sdir_data[npyIndex];
            }
        }
    }

    //这是用来根据flame模型（.npz）的顺序生成mean点的，用于测试loss
    // saveVerticesAsTxt(templateVertices, "../Data/3d/test_mean_new.txt");
    // const std::string targetsFile = "../Data/3d/test_mean_new.txt";


    // 2. 加载目标点 (3 × NUM_MACTHED_POINTS)---3248
    Eigen::MatrixXd targets = loadTargetPoints(targetsFile);

    // 3. 初始化参数
    std::vector<double> shapeParameters(numShapeParameters, 0.0);

    // 5. 构造 Ceres 问题
    ceres::Problem problem;
    problem.AddParameterBlock(shapeParameters.data(), numShapeParameters);

    // 初始化（后脑勺部分）
    int indexIdx = 0; // 用来检索indexList和matched target matrix的idx
    std::ofstream matchedFlameFile("../Data/3d/knn/matchedFlame_" + std::to_string(ITERATION) + ".txt");

    //numVertices = 5023, indexList.size() = 3248;
    for (int vi = 0; vi < numVertices; ++vi) {

        //如果indexList里的所有点都已经计算loss了，跳出for循环
        if (indexIdx >= indexList.size()) break;

        // 如果当前点不是 indexList 中的目标点，跳过
        if(indexList[indexIdx] != vi) continue;

        //我担心flame[indexList[indexIdx]]!=flame.txt[indexIdx]
        //记录一下所有没有被跳过的点，看是否跟flame.txt里匹配
        matchedFlameFile << templateVertices(vi, 0) << " " << templateVertices(vi, 1) << " " << templateVertices(vi, 2)<< "\n";

        // P2P loss
        auto* cost_p2p = new ceres::DynamicAutoDiffCostFunction<P2PointResidual>(
            new P2PointResidual(vi, targets.col(indexIdx).eval())
        );
        cost_p2p->AddParameterBlock(numShapeParameters);
        cost_p2p->SetNumResiduals(3);
        problem.AddResidualBlock(cost_p2p, nullptr, shapeParameters.data());
        
        
        //一共3248个匹配点，所以indexIdx最后应该等于3248（从0开始）
        //好吧，每次knn完后数量不确定，但应该等于NUM_MACTHED_POINTS
        indexIdx++; 

    }

    // 6. 求解
    // 优化器设置是直接照抄exercise5里的设置
    ceres::Solver::Options opts;
    opts.trust_region_strategy_type  = ceres::LEVENBERG_MARQUARDT;
    opts.use_nonmonotonic_steps       = false;
    opts.linear_solver_type           = ceres::DENSE_QR;
    opts.minimizer_progress_to_stdout = 1;
    opts.num_threads                  = 8;

    ceres::Solver::Summary summary;
    ceres::Solve(opts, &problem, &summary);
    std::cout << summary.FullReport() << std::endl;


    //关闭
    matchedFlameFile.close();
    // 8. 保存 betas
    std::ofstream betaFile("../Data/3d/knn/test_betas_" + std::to_string(ITERATION) + ".txt");
    for (double b : shapeParameters) betaFile << b << "\n";
    betaFile.close();
    std::cout << "Saved shape parameters to test_betas_" + std::to_string(ITERATION) + ".txt\n";
    std::cout << "Indexidx: " << indexIdx << "\n";
    std::cout << "NUM_MACTHED_POINTS: " << NUM_MACTHED_POINTS << "\n";
    std::cout << "如果相等则没问题" << "\n";



    return 0;
}
