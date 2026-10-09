#pragma once
// 单台海康 USB3 工业相机封装（MVS SDK）
// 职责：按序列号打开、配置曝光/增益/帧率/触发、抓帧，返回 BGR 图 + 【设备时间戳】(ns)。
#include <opencv2/core.hpp>
#include <cstdint>
#include <string>

namespace auto_aim
{

class HikCamera
{
  public:
    ~HikCamera();

    // serial：相机序列号（海康 MVS 客户端里可查）；external_trigger：硬触发模式（Line0）
    bool open(const std::string & serial, double exposure_us, double gain, int fps,
              bool external_trigger);

    void close();

    // 抓一帧（阻塞 timeout_ms）。dev_ts_ns=设备时间戳(ns)，frame_num=帧号(用于检测器复用框判断)。
    bool grab(cv::Mat & bgr, std::uint64_t & dev_ts_ns, std::uint32_t & frame_num,
              int timeout_ms = 1000);

    bool isOpen() const { return handle_ != nullptr; }

  private:
    void set_float(const char * name, double value);
    void set_enum(const char * name, unsigned int value);
    void * handle_{nullptr};
};

}  // namespace auto_aim
