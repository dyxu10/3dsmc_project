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
static Eigen::MatrixXd              templateVertices; // mean脸的模版顶点
static std::vector<double>          shapeDirections; // 形变方向
static std::vector<double>          shapeParameters; // 形变参数（betas）
static std::vector<Eigen::Vector3i> faces; // 
static std::vector<Eigen::Vector3d> vertex_normals; // 法线容器（在calculateNorms方法里自动初始化
static std::vector<int>             indexList; // 存放index of matched flame vertices
static int numVertices         = -1;
static int numShapeParameters  = -1;
static int numFaces            = -1;
static int ITERATION           = 1; //用来记录这是第几轮优化（loss+knn算一轮）
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
    P2PointResidual(int vertexIndex, const Eigen::Vector3d &targetPoint, const double weight)
      : vertexIndex_(vertexIndex), targetPoint_(targetPoint), weight_(weight) {}

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
            px += T(shapeDirections[(vertexIndex_ * 3 + 0)*numShapeParameters + k]) * beta;
            py += T(shapeDirections[(vertexIndex_ * 3 + 1)*numShapeParameters + k]) * beta;
            pz += T(shapeDirections[(vertexIndex_ * 3 + 2)*numShapeParameters + k]) * beta;
        }

        residuals[0] = T(weight_) * (px - T(targetPoint_(0)));
        residuals[1] = T(weight_) * (py - T(targetPoint_(1)));
        residuals[2] = T(weight_) * (pz - T(targetPoint_(2)));
        return true;
    }

    int vertexIndex_;
    Eigen::Vector3d targetPoint_;
    double weight_;
};

// 点到面残差
struct P2PlaneResidual {
    P2PlaneResidual(int vertexIndex, const Eigen::Vector3d &targetPoint, const Eigen::Vector3d &normal, const double weight)
      : vertexIndex_(vertexIndex), targetPoint_(targetPoint), normal_(normal), weight_(weight) {}

    template <typename T>
    bool operator()(T const* const* parameters, T* residuals) const {
    
        // 先复用点到点计算(这里设置权重为1.0，因为外围还有点到面权重)
        P2PointResidual p2p(vertexIndex_, targetPoint_, 1.0);
        T p2pt[3];
        p2p(parameters, p2pt);

        // 点到面残差 = (p - q)·n
        residuals[0] = T(weight_) * (T(normal_.x()) * p2pt[0] + T(normal_.y()) * p2pt[1] + T(normal_.z()) * p2pt[2]);
        return true;
    }

    int vertexIndex_;
    Eigen::Vector3d targetPoint_;
    Eigen::Vector3d normal_;
    double weight_;
};

// 计算顶点法线（自动初始化法向量容器）
template <typename T>
void calculateNormals(const T* shapeParams, std::vector<Eigen::Matrix<T,3,1>>& normals) {
    
    if (numVertices == -1) throw std::runtime_error("Not correctly initialize number of vertices yet."); 

    // 初始化+清零
    if (vertex_normals.size() != numVertices) vertex_normals.resize(numVertices);
    for (auto& n : normals) n.setZero();

    // 对每个三角面
    for (const auto& f : faces) {
        Eigen::Matrix<T,3,3> pts;
        for (int i = 0; i < 3; ++i) {
            int vi = f[i];
            // 顶点坐标 + 形变
            T x = T(templateVertices(vi, 0)),
              y = T(templateVertices(vi, 1)),
              z = T(templateVertices(vi, 2));
            for (int k = 0; k < numShapeParameters; ++k) {
                T beta = shapeParams[k];
                x += T(shapeDirections[(vi * 3 + 0) * numShapeParameters + k]) * beta;
                y += T(shapeDirections[(vi * 3 + 1) * numShapeParameters + k]) * beta;
                z += T(shapeDirections[(vi * 3 + 2) * numShapeParameters + k]) * beta;
            }
            pts.col(i) = Eigen::Matrix<T,3,1>(x,y,z);
        }
        // 面法线
        Eigen::Matrix<T,3,1> fn = (pts.col(1) - pts.col(0)).cross(pts.col(2) - pts.col(0));
        // 累加到对应顶点
        for (int i = 0; i < 3; ++i) normals[f[i]] += fn;
    }
    // 归一化
    for (auto& n : normals) {
        T norm = n.norm();
        if (norm > T(1e-8)) n /= norm;
    }
}



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
    // ------- 1 准备工作 ------- 
    std::cout << "reading the model...";
    // 1.1 读取目标点云，加载 FLAME 模型
    // const std::string input_off = "../model/mesh/00001_transform_onlyface.off";
    const std::string input_off = "../model/mesh/transformed_00028.off";
    // Load target point cloud (transformed points)
    MatrixXf target = load_off_as_matrix(input_off);

    // 1.2 加载flame模型
    const std::string flameModel  = "../model/FLAME2023/face_only_mesh.npz";
    // const std::string flameModel  = "../model/FLAME2023/flame2023_new.npz";
    auto vTpl  = cnpy::npz_load(flameModel, "v_template");
    auto sDirs  = cnpy::npz_load(flameModel, "shapedirs");
    auto fArr   = cnpy::npz_load(flameModel, "f");

    numVertices        = int(vTpl.shape[0]);
    numShapeParameters = int(sDirs.shape[2]);
    numFaces           = int(fArr.shape[0]);


    // ------- 2 初始化 ------- 
    std::cout << "initializing the parameters...";
    // 2.1 初始化模板顶点（double）
    templateVertices.resize(numVertices, 3);
    const double* vtpl_data = vTpl.data<double>();

    for (int i = 0; i < numVertices; ++i) {
        templateVertices(i, 0) = vtpl_data[i * 3 + 0];  // x
        templateVertices(i, 1) = vtpl_data[i * 3 + 1];  // y
        templateVertices(i, 2) = vtpl_data[i * 3 + 2];  // z
    }

    // 2.2 初始化形变方向
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

    // 2.3 初始化betas参数
    shapeParameters.assign(numShapeParameters, 0.0);

    // 2.4 初始化faces
    int* f_data = fArr.data<int>();  // npz 中 f 应当是 int32
    faces.resize(numFaces);
    for (int i = 0; i < numFaces; ++i) {
        faces[i] = Eigen::Vector3i(
            f_data[3*i+0],
            f_data[3*i+1],
            f_data[3*i+2]
        );
    }

    // =============================================================================================================

    while(ITERATION <= MAX_ITERATION){  
        
        // ------- 3 knn ------- 
        std::cout << "now start with "<< ITERATION << "-th iteration of knn.";

        // 3.1 knn(vTpl,sDirs,shapeParameters)
        Flame_Mesh mesh(vTpl,sDirs,shapeParameters);
        KNN_Result knn_result = knn(mesh, target);

        std::cout << "dimensions of source : " << knn_result.source.cols() << std::endl;
        std::cout << "dimensions of nn_points : " << knn_result.nn_points.cols() << std::endl;
        
        // 3.2 update indexList, matchedTargets
        Eigen::MatrixXd matchedTargets = knn_result.nn_points.cast<double>();
        indexList = knn_result.flame_indices;


        // ------- 4 optimization process -------   
        std::cout << "now start with "<< ITERATION << "-th iteration of optimization.";

        // 4.1 构造 Ceres 问题
        ceres::Problem problem;
        problem.AddParameterBlock(shapeParameters.data(), numShapeParameters);

        // 4.2 初始化三个权重
        const double weight_p2point = 0.5;
        const double weight_p2plane = 0.5;
        const double lambda = 1e-5; // lambda越大，每次可变空间越小

        // 4.3 初始化法向量
        calculateNormals<double>(shapeParameters.data(), vertex_normals);


        // 4.4 添加loss
        for (int i = 0; i < indexList.size(); ++i) { // i是matched targets的index； vi是flame的index

            int vi = indexList[i];

            // P2Point loss
            auto* cost_p2p = new ceres::DynamicAutoDiffCostFunction<P2PointResidual>(
                new P2PointResidual(vi, matchedTargets.col(i).eval(), weight_p2point)
            );
            cost_p2p->AddParameterBlock(numShapeParameters);
            cost_p2p->SetNumResiduals(3);
            problem.AddResidualBlock(cost_p2p, nullptr, shapeParameters.data());


            // P2Plane loss
            auto* cost_p2pl = new ceres::DynamicAutoDiffCostFunction<P2PlaneResidual>(
                new P2PlaneResidual(vi, matchedTargets.col(i).eval(), vertex_normals[vi], weight_p2plane)
            );
            cost_p2pl->AddParameterBlock(numShapeParameters);
            cost_p2pl->SetNumResiduals(1);
            problem.AddResidualBlock(cost_p2pl, nullptr, shapeParameters.data());

        }


        // 4.5 添加正则约束束缚形变大小
        auto* regCost = new RegularizationCost(lambda, numShapeParameters);
        auto* regFunc = new ceres::DynamicAutoDiffCostFunction<RegularizationCost>(regCost);
        regFunc->AddParameterBlock(numShapeParameters);
        regFunc->SetNumResiduals(numShapeParameters);
        problem.AddResidualBlock(regFunc, nullptr, shapeParameters.data());


        // 4.6 求解
        // 优化器设置是直接照抄exercise5里的设置
        ceres::Solver::Options opts;
        opts.trust_region_strategy_type  = ceres::LEVENBERG_MARQUARDT;
        opts.use_nonmonotonic_steps       = false;
        opts.linear_solver_type           = ceres::DENSE_QR;
        opts.minimizer_progress_to_stdout = 1;
        opts.num_threads                  = 8;
        // opts.max_num_iterations           =;

        ceres::Solver::Summary summary;
        ceres::Solve(opts, &problem, &summary);
        std::cout << summary.FullReport() << std::endl;


        //  ------- 5 保存betas ------- 
        std::ofstream betaFile("../Data/optimize_test/test_betas_" + std::to_string(ITERATION) + ".txt");
        for (double b : shapeParameters) betaFile << b << "\n";
        betaFile.close();
        std::cout << "Saved shape parameters to test_betas_" + std::to_string(ITERATION) + ".txt\n";

        ITERATION ++;
    }


    return 0;
}
