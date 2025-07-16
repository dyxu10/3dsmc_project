#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <Eigen/Dense>

struct Vertex {
    Eigen::Vector3f position;
    Eigen::Vector4i color; // r, g, b, a
};

struct Face {
    std::vector<int> indices;
};

bool isValidVertex(const Vertex& v) {
    return v.position[2] < 4.3f; 
}

int main() {
    std::ifstream in("../out/output.off");
    if (!in.is_open()) {
        std::cerr << "Failed to open OFF file.\n";
        return 1;
    }

    std::string header;
    std::getline(in, header);
    if (header != "COFF") {
        std::cerr << "Not a valid COFF file.\n";
        return 1;
    }

    int numVertices, numFaces, dummy;
    in >> numVertices >> numFaces >> dummy;

    std::vector<Vertex> allVertices(numVertices);
    for (int i = 0; i < numVertices; ++i) {
        float x, y, z;
        int r, g, b, a;
        in >> x >> y >> z >> r >> g >> b >> a;
        allVertices[i] = {Eigen::Vector3f(x, y, z), Eigen::Vector4i(r, g, b, a)};
    }

    std::vector<Face> allFaces(numFaces);
    for (int i = 0; i < numFaces; ++i) {
        int n;
        in >> n;
        Face f;
        for (int j = 0; j < n; ++j) {
            int idx;
            in >> idx;
            f.indices.push_back(idx);
        }
        allFaces[i] = f;
    }


    std::vector<Vertex> filteredVertices;
    std::unordered_map<int, int> oldToNewIndex;
    for (int i = 0; i < allVertices.size(); ++i) {
        if (isValidVertex(allVertices[i])) {
            oldToNewIndex[i] = filteredVertices.size();
            filteredVertices.push_back(allVertices[i]);
        }
    }

    std::vector<Face> filteredFaces;
    for (const auto& f : allFaces) {
        bool valid = true;
        for (int idx : f.indices) {
            if (oldToNewIndex.find(idx) == oldToNewIndex.end()) {
                valid = false;
                break;
            }
        }
        if (valid) {
            Face newFace;
            for (int idx : f.indices) {
                newFace.indices.push_back(oldToNewIndex[idx]);
            }
            filteredFaces.push_back(newFace);
        }
    }


    std::ofstream out("../out/output_1.off");
    if (!out.is_open()) {
        std::cerr << "Cannot write output file.\n";
        return 1;
    }

    out << "COFF\n";
    out << filteredVertices.size() << " " << filteredFaces.size() << " 0\n";
    for (const auto& v : filteredVertices) {
        out << v.position[0] << " " << v.position[1] << " " << v.position[2] << " ";
        out << v.color[0] << " " << v.color[1] << " " << v.color[2] << " " << v.color[3] << "\n";
    }
    for (const auto& f : filteredFaces) {
        out << f.indices.size();
        for (int idx : f.indices) out << " " << idx;
        out << "\n";
    }

    std::cout << "Saved: output_1.off\n";
    return 0;
}