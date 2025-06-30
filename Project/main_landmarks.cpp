#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include "utils.h"

using namespace std;
using namespace cv;
using namespace Eigen;

// 读取 2D landmarks
static vector<Point2f> ReadLandmarks(const string& path) {
    ifstream file(path);
    vector<Point2f> lm;
    float x,y;
    while (file >> x >> y) lm.emplace_back(x,y);
    return lm;
}

// 写纯坐标 .txt
static void WritePtsTxt(const vector<Vector3f>& pts, const string& path) {
    ofstream out(path);
    for (auto& p : pts)
        out << p.x() << " " << p.y() << " " << p.z() << "\n";
}

// 写 COFF .off
static void WritePtsOFF(const vector<Vector3f>& pts,
                        const vector<Vec3b>& cols,
                        const string& path) {
    ofstream out(path);
    out << "COFF\n" << pts.size() << " 0 0\n";
    for (size_t i=0; i<pts.size(); ++i) {
        auto& p = pts[i];
        auto& c = cols[i];
        out << p.x() << " " << p.y() << " " << p.z() << " "
            << int(c[2]) << " " << int(c[1]) << " " << int(c[0]) << " 255\n";
    }
}

int main(){
    // —— 路径配置 —— 
    const string depthPath  = "../data/depth/00001.png";
    const string colorPath  = "../data/color/00001.png";
    const string lmPath     = "../data/landmarks_2d_text/00001.txt";
    const string dintrPath  = "../data/camera/c00_depth_intrinsic.txt";
    const string dextrPath  = "../data/camera/c00_depth_extrinsic.txt";
    const string cintrPath  = "../data/camera/c00_color_intrinsic.txt";
    const string cextrPath  = "../data/camera/c00_color_extrinsic.txt";

    // —— 读取相机参数 —— 
    Matrix3f Kd = ReadIntrinsics(dintrPath);
    auto    Ed3 = ReadExtrinsics(dextrPath);
    Matrix3f Kc = ReadIntrinsics(cintrPath);
    auto    Ec3 = ReadExtrinsics(cextrPath);

    // 准备世界坐标系变换（与 main.cpp 一致）
    Matrix4f Ed4 = ConvertExtrinsicsToHomogeneous(Ed3);
    Matrix4f Ec4 = ConvertExtrinsicsToHomogeneous(Ec3);
    // Depth → Color/World
    Matrix4f T_color = Ec4 * Ed4.inverse();

    // —— 读取图像与 landmarks —— 
    Mat depth = imread(depthPath, IMREAD_UNCHANGED);
    Mat color = imread(colorPath, IMREAD_COLOR);
    auto lms  = ReadLandmarks(lmPath);

    if(depth.empty()||color.empty()){
        cerr<<"Failed to load images\n"; 
        return -1;
    }

    // 准备容器
    vector<Vector3f> pts_depth, pts_color;
    vector<Vec3b>    cols_depth, cols_color;
    pts_depth.reserve(lms.size());
    pts_color.reserve(lms.size());
    cols_depth.reserve(lms.size());
    cols_color.reserve(lms.size());

    // —— 逐点处理 —— 
    for(size_t i=0; i<lms.size(); ++i){
        int x=int(lms[i].x), y=int(lms[i].y);
        if(x<0||x>=depth.cols||y<0||y>=depth.rows) continue;

        ushort dr = depth.at<ushort>(y,x);
        float   d  = dr / 1000.0f;  // dr==0 会使 d==0

        // 反投影到深度相机坐标系
        float X = (x - Kd(0,2)) * d / Kd(0,0);
        float Y = (y - Kd(1,2)) * d / Kd(1,1);
        float Z = d;
        Vector3f p_d(X,Y,Z);
        pts_depth.push_back(p_d);
        cols_depth.push_back(color.at<Vec3b>(y,x));

        // 如果你想所有点都生成，包括 dr==0，也可以删掉下面判断
        // if(dr==0) continue;

        // 变换到彩色/世界坐标系
        Vector4f p_cam(X,Y,Z,1.0f);
        Vector4f p_col = T_color * p_cam;
        pts_color.emplace_back(p_col.x(), p_col.y(), p_col.z());
        cols_color.push_back(color.at<Vec3b>(y,x));
    }

    // —— 导出四个文件 —— 
    // 1. 深度坐标 txt/off
    WritePtsTxt(pts_depth, "../out/landmarks3D_depth.txt");
    WritePtsOFF(pts_depth, cols_depth, "../out/landmarks3D_depth.off");

    // 2. 彩色/世界坐标 txt/off
    WritePtsTxt(pts_color, "../out/landmarks3D_color.txt");
    WritePtsOFF(pts_color, cols_color, "../out/landmarks3D_color.off");

    cout << "Exported:\n"
         << " - " << pts_depth.size() << " pts to landmarks3D_depth.*\n"
         << " - " << pts_color.size() << " pts to landmarks3D_color.*\n";
    return 0;
}
