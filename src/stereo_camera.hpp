#pragma once
// 双目相机：封装左右两台海康相机，抓一对同步帧（硬触发保证同帧），给出时基。
#include <opencv2/core.hpp>
#include <cstdint>
#include <string>

#include "hik_camera.hpp"

namespace auto_aim
{

struct StereoFrame
{
    cv::Mat left, right;
    double t_capture{0.0};         // 曝光时刻（秒，左相机设备时间戳；单调时基，喂 EKF）
    std::uint64_t frame_id{0};     // 帧号（左相机），单调递增
};

class StereoCamera
{
  public:
    bool open(const std::string & left_serial, const std::string & right_serial,
              double exposure_us, double gain, int fps, bool external_trigger);
    void close();
    // 抓一对：先左后右（硬触发同帧）。t_capture 用左相机设备时间戳。
    bool grab(StereoFrame & f, int timeout_ms = 1000);
    bool isOpen() const;

  private:
    HikCamera left_, right_;
};

}  // namespace auto_aim
