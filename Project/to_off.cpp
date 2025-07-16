//transform txt to off 可视化

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <Eigen/Dense>

struct Vertex {
    Eigen::Vector3f position;
};

int main() {
    std::string input_txt = "../data/flame_mediapipe_landmarks.txt";
    std::string output_off = "../data/flame_mediapipe_landmarks.off";

    std::ifstream in(input_txt);
    if (!in.is_open()) {
        std::cerr << "Failed to open input TXT file: " << input_txt << std::endl;
        return 1;
    }

    std::vector<Vertex> vertices;
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream iss(line);
        float x, y, z;
        if (iss >> x >> y >> z) {
            vertices.push_back({Eigen::Vector3f(x, y, z)});
        }
    }
    in.close();

    std::ofstream out(output_off);
    if (!out.is_open()) {
        std::cerr << "Failed to open output OFF file: " << output_off << std::endl;
        return 1;
    }

    // Write COFF header
    out << "COFF\n";
    out << vertices.size() << " 0 0\n";
    for (const auto& v : vertices) {
        out << v.position[0] << " " << v.position[1] << " " << v.position[2] << " ";
        out << "255 255 255 255\n";  // white with full alpha
    }
    out.close();

    std::cout << "Successfully converted TXT to OFF. Saved to " << output_off << std::endl;
    return 0;
}