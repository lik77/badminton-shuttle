#pragma once
// 双目三角化 + 反投影 + 右图匹配（把左右相机的亚像素中心变成 3D，再反投影回像素喂检测器）
//
// 约定：检测在【校正后】的左右图上做（先 stereoRectify 再检测），
//       triangulate() 输出相机光轴系：+X 右、+Y 下、+Z 深度。
//       （喂 DragPredictor 前要乘固定旋转转到 +Z 上的世界系，见 shuttle_main.example.cpp TODO4）

#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <string>

namespace auto_aim
{

struct StereoParams
{
    double f_px{0.0};      // 校正后焦距(px)：8mm 组≈2319，16mm 组≈4638（按标定填）
    double cx{0.0};        // 校正后主点 x
    double cy{0.0};        // 校正后主点 y
    double baseline{0.0};  // 基线(m)，实测填
    double min_disparity{1.0};    // 视差下限，太小(太远/匹配错)直接拒
    double match_search_px{40.0}; // 右图匹配的水平搜索半宽(px)
};

class StereoRig
{
  public:
    explicit StereoRig(const std::string & config_path);

    // ul/ur：左右校正图上球的亚像素 x；v：极线对齐后的共同 y（两图同 v）
    bool triangulate(double ul, double ur, double v, Eigen::Vector3d & p) const;

    // 3D → 左图像素，用于把 EKF 预测点喂回 ShuttleDetector::detect 的 predicted_px
    cv::Point2f project(const Eigen::Vector3d & p) const;

    // 右图匹配：在右校正图的同 v 行、预期视差位置附近找亮斑，返回亚像素 x（ur）。
    // expected_disparity 用预测的 Z 估：disp = f_px * baseline / Z。找不到返回 NaN。
    double match_right(const cv::Mat & right_gray, double v, double ul,
                       double expected_disparity) const;

    const StereoParams & params() const { return p_; }

  private:
    StereoParams p_;
};

}  // namespace auto_aim
