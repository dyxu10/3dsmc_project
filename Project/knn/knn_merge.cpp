#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Dense>
#include <limits>
#include <omp.h>
#include "cnpy.h"

using namespace std;
using namespace Eigen;


struct Knn_Result{
    Eigen::MatrixXf source;
    Eigen::MatrixXf nn_points;
    std::vector<int> flame_indices;
};


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




void save_matrix_as_txt(const MatrixXf& source, const MatrixXf& nn_points, std::vector<int>& flame_indices){

    //flame.txt : source
    //matched.txt : nn_points
    //indices.txt : index of source

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

KNN_Result knn(const MatrixXf& source){

    //Source : FLAME mesh
    //Target : transformed points(our image point cloud)
    //Source changes after each iteration of optimizer
    //Target is fixed
    //Return : source and nn_points


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
    save_matrix_as_txt(source, nn_points, flame_indices);


    Knn_Result knn_result;
    knn_result.source = source;
    knn_result.nn_points = nn_points;
    knn_result.flame_indices = flame_indices;

    return knn_result;
}

