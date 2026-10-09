#pragma once
// 双组双目系统：管理 8mm(wide) 与 16mm(tele) 两套双目 + 外参 + 坐标系统一。
//
// 坐标系约定（重要）：
//   每组双目三角化输出【光轴系】：+X 右、+Y 下、+Z 深度（前）。
//   EKF 工作在【世界系】：+X 前、+Y 左、+Z 上（重力在 -Z）。
//   所以需要两个固定变换：
//     R_c2w：wide 光轴系 → 世界系（默认把"+Z前/+Y下"转成"+X前/+Z上"，上机后用手眼标定修正俯仰）
//     R_t2w,t_t2w：tele 光轴系 → wide 光轴系（两组外参，标定得到）
#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <memory>
#include <string>

#include "rectifier.hpp"
#include "stereo.hpp"
#include "stereo_camera.hpp"

namespace auto_aim
{

struct RigConfig
{
    std::string name;
    std::string left_serial, right_serial;
    double exposure_us{60.0};
    double gain{10.0};
    int fps{150};
    bool external_trigger{true};
    std::string stereo_yaml;    // 该组的标定参数（configs/stereo_wide.yaml 或 stereo_tele.yaml）
    std::string rectifier_xml;  // 该组的校正映射（stereo_maps_wide.xml 或 _tele.xml）
};

// 一次抓到的两组帧
struct RigFrames
{
    StereoFrame wide, tele;  // wide=8mm 组，tele=16mm 组
};

// 一个双目组：相机 + 校正 + 三角化（按配置组装）
class StereoRigUnit
{
  public:
    bool open(const RigConfig & cfg);
    void close();
    bool grab(StereoFrame & f, int timeout_ms = 1000) { return cam_.grab(f, timeout_ms); }
    // 立体校正左右帧
    bool rectify(const StereoFrame & f, cv::Mat & L, cv::Mat & R) const;
    StereoRig & stereo() { return *rig_; }
    bool ready() const { return cam_.isOpen() && rect_ && rig_ && rect_->ready(); }

  private:
    StereoCamera cam_;
    std::unique_ptr<StereoRectifier> rect_;
    std::unique_ptr<StereoRig> rig_;
};

// 两套双目的管理：打开、抓帧、坐标变换
class RigSystem
{
  public:
    // rigs_yaml：configs/rigs.yaml（两组配置 + 外参 + 交接距离）
    bool load(const std::string & rigs_yaml);
    void close();
    bool grab(RigFrames & f, int timeout_ms = 1000);

    StereoRigUnit & wide() { return wide_; }
    StereoRigUnit & tele() { return tele_; }

    // 光轴系 → 世界系
    Eigen::Vector3d wide_to_world(const Eigen::Vector3d & p) const;
    Eigen::Vector3d tele_to_world(const Eigen::Vector3d & p) const;
    // 世界系 → 光轴系（用于把 EKF 预测点反投影回某组左图）
    Eigen::Vector3d world_to_wide(const Eigen::Vector3d & p) const;
    Eigen::Vector3d world_to_tele(const Eigen::Vector3d & p) const;

    double handoff_dist() const { return handoff_dist_; }

  private:
    StereoRigUnit wide_, tele_;
    Eigen::Matrix3d R_c2w_;   // wide 光轴系 → 世界系
    Eigen::Matrix3d R_t2w_;   // tele 光轴系 → wide 光轴系
    Eigen::Vector3d t_t2w_;
    double handoff_dist_{5.0};  // 超过此距离(米)用 tele 为主，近于用 wide 为主
};

}  // namespace auto_aim
