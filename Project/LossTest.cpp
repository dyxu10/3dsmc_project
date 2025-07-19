// LossTest.cpp


#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <Eigen/Dense>
#include <ceres/ceres.h>
#include "cnpy.h"
#include <limits>
#include <omp.h>
using MatrixXf = Eigen::MatrixXf;
using Vector3f = Eigen::Vector3f;

// —— 全局变量 ——
static Eigen::MatrixXd templateVertices; // mean脸的模版顶点
static std::vector<double>    shapeDirections;  
static std::vector<double> shapeParameters;
static std::vector<Eigen::Vector3i> faces;
static int numVertices        = 0;
static int numShapeParameters = 0;
static int numFaces           = 0;
static std::vector<int> indexList;
static int ITERATION = 1; //用来记录这是第几轮优化（loss+knn算一轮）
static const int MAX_ITERATION = 3; // 设置一共跑几轮


// —— knn用到的结构 ——
struct KNN_Result{
    Eigen::MatrixXf source;
    Eigen::MatrixXf nn_points;
    std::vector<int> flame_indices;
};

struct Flame_Mesh{
    const cnpy::NpyArray& v_template_arr;
    const cnpy::NpyArray& shapedirs_arr;
    const std::vector<double>& betas;

    Flame_Mesh(const cnpy::NpyArray& v, const cnpy::NpyArray& s, const std::vector<double>& b)
    : v_template_arr(v), shapedirs_arr(s), betas(b) {}
};

// β 的正则化残差项
struct RegularizationCost {
    RegularizationCost(double lambda, int n_params)
        : lambda_(lambda), n_params_(n_params) {}

    template <typename T>
    bool operator()(T const* const* parameters, T* residuals) const {
        const T* beta = parameters[0];
        for (int i = 0; i < n_params_; ++i) {
            residuals[i] = T(std::sqrt(lambda_)) * beta[i];
        }
        return true;
    }

private:
    double lambda_;
    int n_params_;
};



// 点到点残差
struct P2PointResidual {
    P2PointResidual(int vertexIndex, const Eigen::Vector3d &targetPoint)
      : vertexIndex_(vertexIndex), targetPoint_(targetPoint) {}

    template <typename T>
    bool operator()(T const* const* parameters, T* residuals) const {
        const T* betas = parameters[0];

        // 模板顶点
        T px = T(templateVertices(vertexIndex_, 0));
        T py = T(templateVertices(vertexIndex_, 1));
        T pz = T(templateVertices(vertexIndex_, 2));

        // 叠加形变
        for (int k = 0; k < numShapeParameters; ++k) {
            T beta = betas[k];
            px += T(shapeDirections[(vertexIndex_*3 + 0)*numShapeParameters + k]) * beta;
            py += T(shapeDirections[(vertexIndex_*3 + 1)*numShapeParameters + k]) * beta;
            pz += T(shapeDirections[(vertexIndex_*3 + 2)*numShapeParameters + k]) * beta;
        }

        residuals[0] = px - T(targetPoint_(0));
        residuals[1] = py - T(targetPoint_(1));
        residuals[2] = pz - T(targetPoint_(2));
        return true;
    }

    int vertexIndex_;
    Eigen::Vector3d targetPoint_;
};



// Apply shape blendshapes: v_template + shapedirs * betas
MatrixXf apply_shape_blendshape(const cnpy::NpyArray& v_template_arr,
                                 const cnpy::NpyArray& shapedirs_arr,
                                 const std::vector<double>& betas) {
    const double* v_data = v_template_arr.data<double>();
    const double* s_data = shapedirs_arr.data<double>();

    int N = v_template_arr.shape[0];
    int B = shapedirs_arr.shape[2];

    MatrixXf vertices(3, N);
    for (int i = 0; i < N; ++i) {
        Vector3f v(static_cast<float>(v_data[i * 3 + 0]),
                   static_cast<float>(v_data[i * 3 + 1]),
                   static_cast<float>(v_data[i * 3 + 2]));
        for (int b = 0; b < B; ++b) {
            v.x() += static_cast<float>(s_data[i * 3 * B + 0 * B + b]) * betas[b];
            v.y() += static_cast<float>(s_data[i * 3 * B + 1 * B + b]) * betas[b];
            v.z() += static_cast<float>(s_data[i * 3 * B + 2 * B + b]) * betas[b];
        }
        vertices.col(i) = v;
    }
    return vertices;
}


void save_matrix_as_txt(const MatrixXf& source, const MatrixXf& nn_points, std::vector<int>& flame_indices){

    //flame.txt : source
    //matched.txt : nn_points
    //indices.txt : index of source

    // Open output files
    std::ofstream flame_out("../Data/optimize_test/flame_" + std::to_string(ITERATION) + ".txt");
    std::ofstream match_out("../Data/optimize_test/matched_" + std::to_string(ITERATION) + ".txt");
    std::ofstream index_out("../Data/optimize_test/indices_" + std::to_string(ITERATION) + ".txt");

        // Configurable distance threshold
    float max_distance = 0.02f;
    int valid_count = 0;
    float total_distance = 0.0f;
    float max_dist_observed = 0.0f;

    // Write filtered matched points and compute distance statistics
    for (int i = 0; i < source.cols(); ++i) {
        float dist = (source.col(i) - nn_points.col(i)).norm();
        if (dist > max_distance) continue;

        flame_out << source(0, i) << " " << source(1, i) << " " << source(2, i) << "\n";
        match_out << nn_points(0, i) << " " << nn_points(1, i) << " " << nn_points(2, i) << "\n";
        index_out << i << "\n";  // Only output FLAME vertex index
        flame_indices.push_back(i);

        // std::cout << "flame_indices[i]: " << flame_indices[i] << std::endl;

        total_distance += dist;
        if (dist > max_dist_observed) max_dist_observed = dist;

        ++valid_count;
    }
    // Print summary
    if (valid_count > 0) {
        float mean_distance = total_distance / valid_count;
        std::cout << "KNN with betas completed. " << valid_count << " valid matches." << std::endl;
        std::cout << "Mean distance: " << mean_distance << std::endl;
        std::cout << "Max distance: " << max_dist_observed << std::endl;
    } else {
        std::cout << "No valid matches found (all distances exceed threshold)." << std::endl;
    }
}

// Load OFF file as 3xN matrix
MatrixXf load_off_as_matrix(const std::string& filename) {
    std::ifstream in(filename);
    if (!in.is_open()) throw std::runtime_error("Cannot open file: " + filename);

    std::string header;
    in >> header;
    if (header != "OFF" && header != "COFF") throw std::runtime_error("Not an OFF/COFF file");

    int numVertices, numFaces, dummy;
    in >> numVertices >> numFaces >> dummy;
    MatrixXf mat(3, numVertices);
    for (int i = 0; i < numVertices; ++i) {
        float x, y, z;
        in >> x >> y >> z;
        mat(0, i) = x;
        mat(1, i) = y;
        mat(2, i) = z;
        if (header == "COFF") { int r, g, b, a; in >> r >> g >> b >> a; }
    }
    return mat;
}

// Parallel KNN search
std::vector<int> knn_search_parallel(const MatrixXf& source, const MatrixXf& target) {
    std::vector<int> nn_indices(source.cols(), -1);
    #pragma omp parallel for
    for (int i = 0; i < source.cols(); ++i) {
        float min_dist = std::numeric_limits<float>::max();
        int min_j = -1;
        for (int j = 0; j < target.cols(); ++j) {
            float dist = (source.col(i) - target.col(j)).squaredNorm();
            if (dist < min_dist) {
                min_dist = dist;
                min_j = j;
            }
        }
        nn_indices[i] = min_j;
    }
    return nn_indices;
}

KNN_Result knn(Flame_Mesh& flame_mesh, const MatrixXf& target){

    //Source : FLAME mesh
    //Target : transformed points(our image point cloud)
    //Source changes after each iteration of optimizer
    //Target is fixed
    //Return : source and nn_points

    // Load FLAME shape model
    cnpy::NpyArray v_template_arr = flame_mesh.v_template_arr;
    cnpy::NpyArray shapedirs_arr = flame_mesh.shapedirs_arr;
    // std::vector<double> betas = flame_mesh.betas;
    const std::vector<double>& betas = flame_mesh.betas;

    MatrixXf source;

    // Add betas to betas_vector
    source = apply_shape_blendshape(v_template_arr, shapedirs_arr, betas);
    

    // Generate FLAME mesh with shape deformation


    // Run parallel KNN matching
    // Source : FLAME mesh
    // Target : transformed points(our image point cloud)
    // knn result : nearest point of source.col(i) in target = target.col(nn_indices[i])
    std::vector<int> nn_indices = knn_search_parallel(source, target);

    // Build matched point set
    // nearest point of source.col(i) in target = nn_points.col(i)
    MatrixXf nn_points(3, source.cols());
    for (int i = 0; i < source.cols(); ++i)
        nn_points.col(i) = target.col(nn_indices[i]);


    std::vector<int> flame_indices;
    // Write filtered matched points and compute distance statistics
    // save_matrix_as_txt(source, nn_points, flame_indices);

    // Apply the same distance filter to create filtered matrices
    float max_distance = 0.02f;
    std::vector<int> valid_indices;
    
    for (int i = 0; i < source.cols(); ++i) {
        float dist = (source.col(i) - nn_points.col(i)).norm();
        if (dist <= max_distance) {
            valid_indices.push_back(i);
        }
    }
    
    // Create filtered matrices with only valid points
    MatrixXf filtered_source(3, valid_indices.size());
    MatrixXf filtered_nn_points(3, valid_indices.size());
    
    for (int i = 0; i < valid_indices.size(); ++i) {
        filtered_source.col(i) = source.col(valid_indices[i]);
        filtered_nn_points.col(i) = nn_points.col(valid_indices[i]);
    }

    // for (auto i : flame_indices){
    //     std::cout << "flame_indices[i]: " << i << std::endl;
    // }

    KNN_Result knn_result;
    knn_result.source = filtered_source;
    knn_result.nn_points = filtered_nn_points;
    knn_result.flame_indices = valid_indices;

    return knn_result;
}



int main() {
    // 1. 读取目标点云，加载 FLAME 模型
    const std::string input_off = "../model/mesh/00001_transform_onlyface.off";
    // Load target point cloud (transformed points)
    MatrixXf target = load_off_as_matrix(input_off);

    const std::string flameModel  = "../model/FLAME2023/flame2023_new.npz";
    auto vTpl  = cnpy::npz_load(flameModel, "v_template");
    auto sDirs  = cnpy::npz_load(flameModel, "shapedirs");
    // auto fArr   = cnpy::npz_load(flameModel, "faces");

    numVertices        = int(vTpl.shape[0]);
    numShapeParameters = int(sDirs.shape[2]);
    // numFaces           = int(fArr.shape[0]);

    // 初始化模板顶点（double）
    templateVertices.resize(numVertices, 3);
    const double* vtpl_data = vTpl.data<double>();

    for (int i = 0; i < numVertices; ++i) {
        templateVertices(i, 0) = vtpl_data[i * 3 + 0];  // x
        templateVertices(i, 1) = vtpl_data[i * 3 + 1];  // y
        templateVertices(i, 2) = vtpl_data[i * 3 + 2];  // z
    }

    // 初始化形变方向
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

    // 2. 初始化betas参数
    shapeParameters.assign(numShapeParameters, 0.0);

    while(ITERATION <= MAX_ITERATION){   
        std::cout << "now start with "<< ITERATION << "-th iteration of knn.";
        // 3.1 knn(vTpl,sDirs,shapeParameters)
        Flame_Mesh mesh(vTpl,sDirs,shapeParameters);
        KNN_Result knn_result = knn(mesh, target);

        std::cout << "dimensions of source : " << knn_result.source.cols() << std::endl;
        std::cout << "dimensions of nn_points : " << knn_result.nn_points.cols() << std::endl;
        
        // 3.2 update indexList, targetsmatrix
        // 加载目标点
        Eigen::MatrixXd matchedTargets = knn_result.nn_points.cast<double>();

        // update indexList
        indexList = knn_result.flame_indices;


        // 4. optimization process    
        std::cout << "now start with "<< ITERATION << "-th iteration of optimization.";

        // 4.1 构造 Ceres 问题
        ceres::Problem problem;
        problem.AddParameterBlock(shapeParameters.data(), numShapeParameters);

        for (int i = 0; i < indexList.size(); ++i) {
            int vi = indexList[i];
            auto* cost_p2p = new ceres::DynamicAutoDiffCostFunction<P2PointResidual>(
                new P2PointResidual(vi, matchedTargets.col(i).eval())
            );
            cost_p2p->AddParameterBlock(numShapeParameters);
            cost_p2p->SetNumResiduals(3);
            problem.AddResidualBlock(cost_p2p, nullptr, shapeParameters.data());
        }

        const double lambda = 1e-5; // 0.1
        auto* regCost = new RegularizationCost(lambda, numShapeParameters);

        auto* regFunc = new ceres::DynamicAutoDiffCostFunction<RegularizationCost>(regCost);
        regFunc->AddParameterBlock(numShapeParameters);
        regFunc->SetNumResiduals(numShapeParameters);
        problem.AddResidualBlock(regFunc, nullptr, shapeParameters.data());

        // 4.2. 求解
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


        // 5. 保存 betas
        std::ofstream betaFile("../Data/optimize_test/test_betas_" + std::to_string(ITERATION) + ".txt");
        for (double b : shapeParameters) betaFile << b << "\n";
        betaFile.close();
        std::cout << "Saved shape parameters to test_betas_" + std::to_string(ITERATION) + ".txt\n";

        ITERATION ++;
    }


    return 0;
}
