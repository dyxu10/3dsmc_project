// scale down our 3d mesh


#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>
#include <Eigen/Dense>

struct Vertex {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::Vector3f position;
    Eigen::Vector4i color;
};

// 读取 landmarks.txt，每行3列：x y z
Eigen::MatrixXf LoadLandmarks(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) throw std::runtime_error("Cannot open file: " + filename);

    std::vector<Eigen::Vector3f> points;
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        float x, y, z;
        if (iss >> x >> y >> z) {
            points.emplace_back(x, y, z);
        }
    }

    Eigen::MatrixXf result(points.size(), 3);
    for (size_t i = 0; i < points.size(); ++i) {
        result.row(i) = points[i];
    }
    return result;
}

// 中心化 + 均方根 norm
float ComputeRMSNorm(const Eigen::MatrixXf& pts) {
    Eigen::MatrixXf centered = pts.rowwise() - pts.colwise().mean();
    float norm = std::sqrt((centered.array().square().rowwise().sum().mean()));
    return norm;
}

// OFF 文件处理，保留颜色信息
void ReadCOFF(const std::string& filename,
              std::vector<Vertex>& vertices,
              std::vector<std::vector<int>>& faces) {
    std::ifstream in(filename);
    if (!in.is_open()) throw std::runtime_error("Cannot open COFF: " + filename);

    std::string header;
    std::getline(in, header);
    if (header != "COFF")
        throw std::runtime_error("Expected COFF format");

    int numVertices, numFaces, dummy;
    in >> numVertices >> numFaces >> dummy;

    vertices.resize(numVertices);
    for (int i = 0; i < numVertices; ++i) {
        float x, y, z;
        int r, g, b, a;
        in >> x >> y >> z >> r >> g >> b >> a;
        vertices[i].position = Eigen::Vector3f(x, y, z);
        vertices[i].color = Eigen::Vector4i(r, g, b, a);
    }

    faces.resize(numFaces);
    for (int i = 0; i < numFaces; ++i) {
        int n;
        in >> n;
        std::vector<int> f(n);
        for (int j = 0; j < n; ++j) in >> f[j];
        faces[i] = f;
    }
}

void WriteCOFF(const std::string& filename,
               const std::vector<Vertex>& vertices,
               const std::vector<std::vector<int>>& faces) {
    std::ofstream out(filename);
    if (!out.is_open()) throw std::runtime_error("Cannot write COFF: " + filename);

    out << "COFF\n";
    out << vertices.size() << " " << faces.size() << " 0\n";
    for (const auto& v : vertices) {
        out << v.position[0] << " " << v.position[1] << " " << v.position[2] << " ";
        out << v.color[0] << " " << v.color[1] << " " << v.color[2] << " " << v.color[3] << "\n";
    }
    for (const auto& f : faces) {
        out << f.size();
        for (int idx : f) out << " " << idx;
        out << "\n";
    }
}

int main() {
    try {
        // 1. Load landmarks
        Eigen::MatrixXf your_lms = LoadLandmarks("../data/Mediapip_source.txt");
        Eigen::MatrixXf flame_lms = LoadLandmarks("../data/flame_mediapipe_landmarks.txt");
        if (your_lms.rows() != flame_lms.rows())
            throw std::runtime_error("Landmark counts do not match");

        // 2. Compute scale
        float norm_your = ComputeRMSNorm(your_lms);
        float norm_flame = ComputeRMSNorm(flame_lms);
        float scale = norm_flame / norm_your;

        std::cout << "Estimated scale: " << scale << std::endl;

        // 3. Read COFF
        std::vector<Vertex> vertices;
        std::vector<std::vector<int>> faces;
        ReadCOFF("../out/output_1.off", vertices, faces);


        // 4. Apply scale to vertex positions only
        for (auto& v : vertices) {
            v.position *= scale;
        }

        // 5. Save scaled COFF
        WriteCOFF("../out/scaled_output_1.off", vertices, faces);
        std::cout << "Mesh scaled and saved as scaled.off\n";



         // 6. Scale landmarks as well
         for (int i = 0; i < your_lms.rows(); ++i) {
            your_lms.row(i) *= scale;
        } 

        // 7. Save scaled landmarks
         std::ofstream out_lms("../out/scaled_mediapip_matching_rgb.txt");
        if (!out_lms.is_open()) throw std::runtime_error("Cannot write landmarks");
        for (int i = 0; i < your_lms.rows(); ++i) {
            out_lms << your_lms(i, 0) << " " << your_lms(i, 1) << " " << your_lms(i, 2) << "\n";
        }
        out_lms.close();
        std::cout << "Landmarks scaled and saved as scaled_pip_matching.txt\n";
 
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}