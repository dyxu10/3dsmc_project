#include <fstream>
#include <sstream>
#include <Eigen/Dense>
#include <string>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <cmath>

struct Vertex {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::Vector3f position;
    Eigen::Vector4i color;
};

struct Face {
    std::vector<int> indices;
};

void LoadRTFromTXT(const std::string& filename, Eigen::Matrix3f& R, Eigen::Vector3f& T) {
    std::ifstream file(filename);
    if (!file.is_open()) throw std::runtime_error("can't open: " + filename);

    std::string line;
    int row = 0;
    bool readingR = false, readingT = false;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        if (line.find("R:") != std::string::npos) {
            readingR = true;
            row = 0;
            continue;
        } else if (line.find("T:") != std::string::npos) {
            readingR = false;
            readingT = true;
            continue;
        }

        if (readingR && row < 3) {
            float r0, r1, r2;
            if (iss >> r0 >> r1 >> r2) {
                R(row, 0) = r0;
                R(row, 1) = r1;
                R(row, 2) = r2;
                row++;
            }
        } else if (readingT) {
            float t0, t1, t2;
            if (iss >> t0 >> t1 >> t2) {
                T << t0, t1, t2;
                break;
            }
        }
    }
}

void ReadOFF(const std::string& filename,
             std::vector<Vertex>& vertices,
             std::vector<Face>& faces) {
    std::ifstream in(filename);
    if (!in.is_open()) throw std::runtime_error("Cannot open OFF file: " + filename);

    std::string header;
    std::getline(in, header);
    if (header != "COFF" && header != "OFF")
        throw std::runtime_error("Not a valid OFF file.");

    int numVertices, numFaces, dummy;
    in >> numVertices >> numFaces >> dummy;

    vertices.resize(numVertices);
    for (int i = 0; i < numVertices; ++i) {
        float x, y, z;
        int r, g, b, a;
        in >> x >> y >> z >> r >> g >> b >> a;
        vertices[i] = {Eigen::Vector3f(x, y, z), Eigen::Vector4i(r, g, b, a)};
    }

    faces.resize(numFaces);
    for (int i = 0; i < numFaces; ++i) {
        int n;
        in >> n;
        faces[i].indices.resize(n);
        for (int j = 0; j < n; ++j) {
            in >> faces[i].indices[j];
        }
    }
}

void ApplyRT(std::vector<Vertex>& vertices, const Eigen::Matrix3f& R, const Eigen::Vector3f& T) {
    for (auto& v : vertices) {
        v.position = R * v.position + T;
    }
}

void WriteOFF(const std::string& filename,
              const std::vector<Vertex>& vertices,
              const std::vector<Face>& faces) {
    std::ofstream out(filename);
    if (!out.is_open()) throw std::runtime_error("Cannot write to: " + filename);

    out << "COFF\n";
    out << vertices.size() << " " << faces.size() << " 0\n";
    for (const auto& v : vertices) {
        out << v.position[0] << " " << v.position[1] << " " << v.position[2] << " ";
        out << v.color[0] << " " << v.color[1] << " " << v.color[2] << " " << v.color[3] << "\n";
    }
    for (const auto& f : faces) {
        out << f.indices.size();
        for (int idx : f.indices) {
            out << " " << idx;
        }
        out << "\n";
    }
}

int main() {
    try {
        Eigen::Matrix3f R;
        Eigen::Vector3f T;
        LoadRTFromTXT("../build/RT_result.txt", R, T);

        std::vector<Vertex> vertices;
        std::vector<Face> faces;

        ReadOFF("../out/scaled_output_1.off", vertices, faces);
        ApplyRT(vertices, R, T);
        WriteOFF("../out/output_transformed_1.off", vertices, faces);

        std::cout << "Transform successful: output_transformed.off\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}