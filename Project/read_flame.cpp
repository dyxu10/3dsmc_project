#include <iostream>
#include <vector>
#include <string>
#include <Eigen/Dense>
#include <fstream>
#include "cnpy.h"
#include <random>

// Read flame from npz file and exports to obj file. Generates random face if GENERATE_RANDOM_FACE set to true, generic face otherwise.

const bool GENERATE_RANDOM_FACE = true;


// Save vertices & faces to OBJ
void save_obj(const std::string& path,
              const std::vector<Eigen::Vector3f>& vertices,
              const std::vector<Eigen::Vector3i>& faces) {
    std::ofstream out(path);
    if (!out.is_open()) {
        std::cerr << "Could not open file for writing: " << path << std::endl;
        return;
    }

    for (const auto& v : vertices) {
        out << "v " << v.x() << " " << v.y() << " " << v.z() << "\n";
    }

    for (const auto& f : faces) {
        out << "f " << (f.x() + 1) << " " << (f.y() + 1) << " " << (f.z() + 1) << "\n";
    }

    out.close();
    std::cout << "Exported mesh to " << path << std::endl;
}

int main() {
    std::string npz_path = "../model/FLAME2023/flame2023_no_jaw.npz";

    // Load v_template
    cnpy::NpyArray v_template_arr = cnpy::npz_load(npz_path, "v_template");
    if (v_template_arr.shape.size() != 2 || v_template_arr.shape[1] != 3) {
        std::cerr << "Unexpected v_template shape!" << std::endl;
        return 1;
    }

    size_t num_vertices = v_template_arr.shape[0];
    const double* v_data = v_template_arr.data<double>();

    // Load faces
    cnpy::NpyArray faces_arr = cnpy::npz_load(npz_path, "faces");
    if (faces_arr.shape.size() != 2 || faces_arr.shape[1] != 3) {
        std::cerr << "Unexpected faces shape!" << std::endl;
        return 1;
    }
    size_t num_faces = faces_arr.shape[0];
    const uint32_t* f_data = faces_arr.data<uint32_t>();

    // Load shapedirs
    cnpy::NpyArray shapedirs_arr = cnpy::npz_load(npz_path, "shapedirs");
    if (shapedirs_arr.shape.size() != 3 || shapedirs_arr.shape[1] != 3) {
        std::cerr << "Unexpected shapedirs shape!" << std::endl;
        return 1;
    }
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
    std::vector<double> betas(num_betas);
    for (size_t i = 0; i < num_betas; ++i) {
        betas[i] =normal_dist(rng);  // random ~ N(0,1)
    }


    // Compute v_shaped = v_template + sum_i shapedirs[:, :, i] * betas[i]
    std::vector<Eigen::Vector3f> vertices(num_vertices);
    if (GENERATE_RANDOM_FACE){

        for (size_t v = 0; v < num_vertices; ++v) {
            Eigen::Vector3d v_shaped(v_data[v * 3 + 0],
                                    v_data[v * 3 + 1],
                                    v_data[v * 3 + 2]);
            for (size_t i = 0; i < num_betas; ++i) {
                v_shaped.x() += s_data[v * 3 * num_betas + 0 * num_betas + i] * betas[i];
                v_shaped.y() += s_data[v * 3 * num_betas + 1 * num_betas + i] * betas[i];
                v_shaped.z() += s_data[v * 3 * num_betas + 2 * num_betas + i] * betas[i];
            }
            vertices[v] = v_shaped.cast<float>();
        }
        std::cout << "Generated deformed mesh with random betas." << std::endl;
    } else {
    // If random face not enabled, use v_template directly
    for (size_t v = 0; v < num_vertices; ++v) {
        vertices[v] = Eigen::Vector3f(static_cast<float>(v_data[v * 3 + 0]),
                                        static_cast<float>(v_data[v * 3 + 1]),
                                        static_cast<float>(v_data[v * 3 + 2]));
        }
        std::cout << "Exporting neutral v_template mesh." << std::endl;
    }

    save_obj("../model/mesh/flame2023_random.obj", vertices, faces);

    return 0;
}
