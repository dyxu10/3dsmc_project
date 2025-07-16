#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Dense>
#include <limits>
#include <omp.h>
#include "cnpy.h"

using namespace std;
using namespace Eigen;

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

// Try load betas file (returns 300x1 vector or zeros if not found)
VectorXd load_betas(const std::string& filepath, size_t num_betas) {
    VectorXd betas = VectorXd::Zero(num_betas);
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cout << "Betas file not found, using zeros." << std::endl;
        return betas;
    }
    for (size_t i = 0; i < num_betas && file >> betas(i); ++i);
    return betas;
}

// Apply shape blendshapes: v_template + shapedirs * betas
MatrixXf apply_shape_blendshape(const cnpy::NpyArray& v_template_arr,
                                 const cnpy::NpyArray& shapedirs_arr,
                                 const VectorXd& betas) {
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
            v.x() += static_cast<float>(s_data[i * 3 * B + 0 * B + b]) * betas(b);
            v.y() += static_cast<float>(s_data[i * 3 * B + 1 * B + b]) * betas(b);
            v.z() += static_cast<float>(s_data[i * 3 * B + 2 * B + b]) * betas(b);
        }
        vertices.col(i) = v;
    }
    return vertices;
}

int main() {
    std::string input_off = "../data/00001_transform_onlyface.off";
    std::string npz_path = "../model/FLAME2023/flame2023_no_jaw.npz";
    std::string beta_path = "../optimizer/test_betas_1.txt";

    // Load target point cloud (transformed points)
    MatrixXf target = load_off_as_matrix(input_off);

    // Load FLAME shape model
    cnpy::NpyArray v_template_arr = cnpy::npz_load(npz_path, "v_template");
    cnpy::NpyArray shapedirs_arr = cnpy::npz_load(npz_path, "shapedirs");

    // Load betas (use zeros if not found)
    size_t num_betas = shapedirs_arr.shape[2];
    VectorXd betas = load_betas(beta_path, num_betas);

    // Generate FLAME mesh with shape deformation
    MatrixXf source = apply_shape_blendshape(v_template_arr, shapedirs_arr, betas);

    // Run parallel KNN matching
    std::vector<int> nn_indices = knn_search_parallel(source, target);

    // Build matched point set
    MatrixXf nn_points(3, source.cols());
    for (int i = 0; i < source.cols(); ++i)
        nn_points.col(i) = target.col(nn_indices[i]);

    // Open output files
    std::ofstream flame_out("../optimizer/flame.txt");
    std::ofstream match_out("../optimizer/matched.txt");
    std::ofstream index_out("../optimizer/indices.txt");

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

    return 0;
}
