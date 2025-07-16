#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Dense>
#include <limits>

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


int main() {
    std::string input_off = "../data/face/00001_transform_onlyface.off";
    // std::string output_txt = "../data/face/00001_pointcloud_matrix.txt";
    MatrixXf face = load_off_as_matrix(input_off);
    std::cout << "Loaded matrix of size: " << face.rows() << "x" << face.cols() << std::endl;

    std::string mm_head_path = "../model/mesh/flame2023_no_jaw.obj";
    MatrixXf mm_head = load_obj_as_matrix(mm_head_path);
    std::cout << "mm_head: " << mm_head.rows() << "x" << mm_head.cols() << std::endl;

    std::vector<int> nn_indices = knn_search(mm_head, face); // mm_head: 3xN1, face: 3xN2

    for (int i = 0; i < mm_head.cols(); ++i) {
        int nn_idx = nn_indices[i]; // index in face
        float distance = (mm_head.col(i) - face.col(nn_idx)).norm();
        std::cout << "Nearest neighbor of point " << i << " in mm_head: " << nn_idx
                  << " distance: " << distance << std::endl;
    }

    // save_matrix(mat, output_txt);
    // std::cout << "Saved matrix to " << output_txt << std::endl;
    return 0;
}
