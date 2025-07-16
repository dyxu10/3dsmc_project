#include <iostream>
#include <vector>
#include <string>
#include <Eigen/Dense>
#include <fstream>
#include "cnpy.h"

// Utility to load knn_searched.off as a 3xN Eigen matrix
Eigen::MatrixXf load_off_points(const std::string& filename) {
    std::ifstream in(filename);
    if (!in.is_open()) throw std::runtime_error("Cannot open file: " + filename);
    std::string header;
    std::getline(in, header); // OFF
    int numVertices, numFaces, numEdges;
    in >> numVertices >> numFaces >> numEdges;
    std::string dummy;
    std::getline(in, dummy); // consume rest of header line
    Eigen::MatrixXf mat(3, numVertices);
    for (int i = 0; i < numVertices; ++i) {
        float x, y, z;
        in >> x >> y >> z;
        mat(0, i) = x;
        mat(1, i) = y;
        mat(2, i) = z;
    }
    return mat;
}

int main() {
    // === Load MM face mesh from npz ===
    std::string npz_path = "../model/FLAME2023/flame2023_no_jaw.npz";
    cnpy::NpyArray v_template_arr = cnpy::npz_load(npz_path, "v_template");
    if (v_template_arr.shape.size() != 2 || v_template_arr.shape[1] != 3) {
        std::cerr << "Unexpected v_template shape!" << std::endl;
        return 1;
    }
    size_t num_vertices = v_template_arr.shape[0];
    const double* v_data = v_template_arr.data<double>();
    Eigen::MatrixXf mm_face(3, num_vertices);
    for (size_t v = 0; v < num_vertices; ++v) {
        mm_face(0, v) = static_cast<float>(v_data[v * 3 + 0]);
        mm_face(1, v) = static_cast<float>(v_data[v * 3 + 1]);
        mm_face(2, v) = static_cast<float>(v_data[v * 3 + 2]);
    }
    std::cout << "Loaded MM face mesh: 3x" << mm_face.cols() << std::endl;

    // === Load kNN points from OFF ===
    std::string knn_off_path = "knn_searched.off";
    Eigen::MatrixXf knn_points = load_off_points(knn_off_path);
    std::cout << "Loaded kNN points: 3x" << knn_points.cols() << std::endl;

    // === Check correspondence ===
    if (mm_face.cols() != knn_points.cols()) {
        std::cerr << "Mismatch: MM face has " << mm_face.cols() << " points, kNN has " << knn_points.cols() << std::endl;
        return 1;
    }
    std::cout << "Both matrices have " << mm_face.cols() << " points. Ready for optimization." << std::endl;

    return 0;
}
