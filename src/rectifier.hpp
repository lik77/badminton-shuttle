#pragma once
// 立体校正器：加载标定脚本导出的 remap 映射，对左右帧做立体校正（极线对齐）。
#include <opencv2/core.hpp>
#include <string>

namespace auto_aim
{

class StereoRectifier
{
  public:
    // maps_xml：calibrate_stereo.py 导出的 stereo_maps.xml（cv::FileStorage）
    explicit StereoRectifier(const std::string & maps_xml);

    bool ready() const { return ready_; }
    void rectify(const cv::Mat & L, const cv::Mat & R, cv::Mat & Lrect, cv::Mat & Rrect) const;

  private:
    bool ready_{false};
    cv::Mat mapl1_, mapl2_, mapr1_, mapr2_;
};

}  // namespace auto_aim
