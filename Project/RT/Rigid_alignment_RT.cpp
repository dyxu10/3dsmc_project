//Rigid Alignment(According to RT)

#include <Eigen/Dense>
#include <iostream>
#include <fstream> 
#include <sstream>    
#include <vector>        


void rigid_alignment(const Eigen::MatrixXd& source, const Eigen::MatrixXd& target,
                     Eigen::Matrix3d& R, Eigen::Vector3d& T) {
    assert(source.rows() == target.rows() && source.cols() == 3 && target.cols() == 3);
    int N = source.rows();

    Eigen::Vector3d source_mean = source.colwise().mean();
    Eigen::Vector3d target_mean = target.colwise().mean();

    Eigen::MatrixXd source_centered = source.rowwise() - source_mean.transpose();
    Eigen::MatrixXd target_centered = target.rowwise() - target_mean.transpose();

    Eigen::Matrix3d H = source_centered.transpose() * target_centered;

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();

    R = V * U.transpose();

    if (R.determinant() < 0) {
        V.col(2) *= -1;
        R = V * U.transpose();
    }

    T = target_mean - R * source_mean;
}

Eigen::MatrixXd ReadPointsFromTXT(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    std::vector<Eigen::Vector3d> points;
    std::string line;

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        double x, y, z;
        if (iss >> x >> y >> z) {
            points.emplace_back(x, y, z);
        }
    }

    Eigen::MatrixXd mat(points.size(), 3);
    for (size_t i = 0; i < points.size(); ++i) {
        mat.row(i) = points[i];
    }

    return mat;
}

Eigen::MatrixXd ReadPointsFromOFF(const std::string& filename) {
    std::ifstream in(filename);
    if (!in.is_open()) {
        throw std::runtime_error("Cannot open OFF file: " + filename);
    }

    std::string header;
    std::getline(in, header);
    if (header != "OFF" && header != "COFF") {
        throw std::runtime_error("Invalid OFF/COFF header in: " + filename);
    }

    int numVertices, numFaces, dummy;
    in >> numVertices >> numFaces >> dummy;

    Eigen::MatrixXd points(numVertices, 3);
    for (int i = 0; i < numVertices; ++i) {
        double x, y, z;
        in >> x >> y >> z;
        points(i, 0) = x;
        points(i, 1) = y;
        points(i, 2) = z;

        if (header == "COFF") {
            int r, g, b, a;
            in >> r >> g >> b >> a;  // skip color
        }
    }

    return points;
}

int main() {
    
    Eigen::MatrixXd source = ReadPointsFromTXT("../data/Landmark/scaled_mediapip_source.txt");
    Eigen::MatrixXd target = ReadPointsFromTXT("../data/Landmark/flame_mediapipe_landmarks.txt");

    Eigen::Matrix3d R;
    Eigen::Vector3d T;

    rigid_alignment(source, target, R, T);

    std::cout << "Rotation:\n" << R << std::endl;
    std::cout << "Translation:\n" << T.transpose() << std::endl;

    std::ofstream outfile("RT_result.txt");
    if (outfile.is_open()) {
        outfile << "R:\n" << R << "\n";
        outfile << "T:\n" << T.transpose() << "\n";
        outfile.close();
        std::cout << "RT successfully written to RT_result.txt" << std::endl;
    } else {
        std::cerr << "Failed to open RT_result.txt for writing." << std::endl;
    }

    return 0;
}