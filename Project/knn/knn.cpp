#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Dense>
#include <limits>
#include <omp.h>
#include <fstream>
#include "cnpy.h"
#include <random>

using namespace std;
using namespace Eigen;

MatrixXf load_off_as_matrix(const std::string& filename) {
    std::ifstream in(filename);
    if (!in.is_open()) throw std::runtime_error("Cannot open file: " + filename);

    std::string header;
    in >> header;
    if (header != "OFF" && header != "COFF") throw std::runtime_error("Not an OFF/COFF file");

    int numVertices, numFaces, dummy;
    in >> numVertices >> numFaces >> dummy;
    std::cout << "numVertices: " << numVertices << std::endl;

    MatrixXf mat(3, numVertices);
    for (int i = 0; i < numVertices; ++i) {
        float x, y, z;
        in >> x >> y >> z;
        mat(0, i) = x;
        mat(1, i) = y;
        mat(2, i) = z;
        // skip color if COFF
        if (header == "COFF") {
            int r, g, b, a;
            in >> r >> g >> b >> a;
        }
    }
    return mat;
}

MatrixXf load_obj_as_matrix(const std::string& filename) {
    std::ifstream in(filename);
    if (!in.is_open()) throw std::runtime_error("Cannot open file: " + filename);

    std::vector<Vector3f> vertices;
    std::string line;
    while (std::getline(in, line)) {
        if (line.size() > 1 && line[0] == 'v' && line[1] == ' ') {
            std::istringstream iss(line.substr(2));
            float x, y, z;
            iss >> x >> y >> z;
            vertices.emplace_back(x, y, z);
        }
    }
    MatrixXf mat(3, vertices.size());
    for (size_t i = 0; i < vertices.size(); ++i) {
        mat.col(i) = vertices[i];
    }
    return mat;
}

void save_matrix(const MatrixXf& mat, const std::string& filename) {
    std::ofstream out(filename);
    if (!out.is_open()) throw std::runtime_error("Cannot open output file");
    out << mat.rows() << " " << mat.cols() << "\n";
    out << mat << "\n";
}

void save_matrix_as_off(const MatrixXf& mat, const std::string& filename) {
    std::ofstream out(filename);
    if (!out.is_open()) throw std::runtime_error("Cannot open output file");

    int numVertices = mat.cols();
    out << "OFF\n";
    out << numVertices << " 0 0\n";
    for (int i = 0; i < numVertices; ++i) {
        out << mat(0, i) << " " << mat(1, i) << " " << mat(2, i) << "\n";
    }
}

// Returns a vector of indices: for each source point, the index of its nearest neighbor in target
std::vector<int> knn_search(const MatrixXf& source, const MatrixXf& target) {
    std::vector<int> nn_indices(source.cols(), -1);

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
        nn_indices[i] = min_j; // index of closest point in target for source.col(i)
    }
    return nn_indices;
}

MatrixXf read_npz_to_matrix(const std::string& npz_path) {
    bool GENERATE_RANDOM_FACE = false;

    

    // Load v_template
    cnpy::NpyArray v_template_arr = cnpy::npz_load(npz_path, "v_template");
 

    size_t num_vertices = v_template_arr.shape[0];
    const double* v_data = v_template_arr.data<double>();

    // Load faces
    cnpy::NpyArray faces_arr = cnpy::npz_load(npz_path, "faces");

    size_t num_faces = faces_arr.shape[0];
    const uint32_t* f_data = faces_arr.data<uint32_t>();

    // Load shapedirs
    cnpy::NpyArray shapedirs_arr = cnpy::npz_load(npz_path, "shapedirs");

    size_t num_betas = shapedirs_arr.shape[2];  // should be 400
    std::cout << "Num of num_betas is " << num_betas << std::endl;
    const double* s_data = shapedirs_arr.data<double>();


    //Prepare faces
    std::vector<Eigen::Vector3i> faces(num_faces);
    for (size_t i = 0; i < num_faces; ++i) {
        faces[i] = Eigen::Vector3i(static_cast<int>(f_data[i * 3 + 0]),
                                   static_cast<int>(f_data[i * 3 + 1]),
                                   static_cast<int>(f_data[i * 3 + 2]));
    }

    // Generate random betas
    // Create a normal distribution with mean=0, stddev=1 (same as np.random.randn / chumpy)
    std::default_random_engine rng(std::random_device{}());
    std::normal_distribution<double> normal_dist(0.0, 1.0);
    std::vector<Eigen::Vector3f> vertices(num_vertices);
     
    // If random face not enabled, use v_template directly
    for (size_t v = 0; v < num_vertices; ++v) {
        vertices[v] = Eigen::Vector3f(static_cast<float>(v_data[v * 3 + 0]),
                                        static_cast<float>(v_data[v * 3 + 1]),
                                        static_cast<float>(v_data[v * 3 + 2]));
        }
        std::cout << "Exporting neutral v_template mesh." << std::endl;
    


    // === Create 3xN matrix from vertices ===

    // Create a 3 x num_vertices matrix (double precision)
    Eigen::MatrixXd face_points(3, num_vertices);

    for (size_t v = 0; v < num_vertices; ++v) {
        face_points(0, v) = static_cast<double>(vertices[v].x());
        face_points(1, v) = static_cast<double>(vertices[v].y());
        face_points(2, v) = static_cast<double>(vertices[v].z());
    }

    return face_points.cast<float>();
}

int main() {
    std::string input_off = "../data/00001_transform_onlyface.off";
    // std::string output_txt = "../data/face/00001_pointcloud_matrix.txt";
    MatrixXf face = load_off_as_matrix(input_off);
    std::cout << "Loaded matrix of size: " << face.rows() << "x" << face.cols() << std::endl;

    // std::string mm_head_path = "../model/mesh/flame2023_no_jaw.obj";
    // MatrixXf mm_head = load_obj_as_matrix(mm_head_path);
    // std::cout << "mm_head: " << mm_head.rows() << "x" << mm_head.cols() << std::endl;

    std::string mm_head_path = "../model/mesh/flame2023_no_jaw.obj";
    MatrixXf mm_head = read_npz_to_matrix(mm_head_path);
    std::cout << "The dimension of mm_head is: " << mm_head.rows() << "x" << mm_head.cols() << std::endl;

    std::vector<int> nn_indices = knn_search(mm_head, face); // mm_head: 3xN1, face: 3xN2

    for (int i = 0; i < mm_head.cols(); ++i) {
        int nn_idx = nn_indices[i]; // index in face
        float distance = (mm_head.col(i) - face.col(nn_idx)).norm();
        std::cout << "Nearest neighbor of point " << i << " in mm_head: " << nn_idx
                  << " distance: " << distance << std::endl;
    }

    MatrixXf nn_points(3, mm_head.cols());
    for (int i = 0; i < mm_head.cols(); ++i) {
        int nn_idx = nn_indices[i];
        nn_points.col(i) = face.col(nn_idx);
    }

    std::cout << "The dimension of nn_points is: " << nn_points.rows() << "x" << nn_points.cols() << std::endl;
    std::cout << "The dimension of mm_head is: " << mm_head.rows() << "x" << mm_head.cols() << std::endl;

    std::string output_off = "../optimizer/knn_searched.off";
    save_matrix_as_off(nn_points, output_off);
    std::cout << "Saved nearest neighbor points to " << output_off << std::endl;

    // save_matrix(mat, output_txt);
    // std::cout << "Saved matrix to " << output_txt << std::endl;
    return 0;
}
