#include "stereo.hpp"

#include <opencv2/imgproc.hpp>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace auto_aim
{

StereoRig::StereoRig(const std::string & config_path)
{
    auto yaml = YAML::LoadFile(config_path);
    p_.f_px = yaml["f_px"].as<double>();
    p_.cx = yaml["cx"].as<double>();
    p_.cy = yaml["cy"].as<double>();
    p_.baseline = yaml["baseline"].as<double>();
    p_.min_disparity = yaml["min_disparity"].as<double>();
    p_.match_search_px = yaml["match_search_px"].as<double>();
}

bool StereoRig::triangulate(double ul, double ur, double v, Eigen::Vector3d & p) const
{
    const double d = ul - ur;  // 视差
    if (d < p_.min_disparity)
    {
        return false;  // 太远 / 左右匹配错
    }
    const double Z = p_.f_px * p_.baseline / d;
    p.x() = (ul - p_.cx) * Z / p_.f_px;
    p.y() = (v - p_.cy) * Z / p_.f_px;
    p.z() = Z;
    return true;
}

cv::Point2f StereoRig::project(const Eigen::Vector3d & p) const
{
    if (p.z() < 1e-6)
    {
        return {0.0F, 0.0F};
    }
    return {static_cast<float>(p_.cx + p_.f_px * p.x() / p.z()),
            static_cast<float>(p_.cy + p_.f_px * p.y() / p.z())};
}

double StereoRig::match_right(const cv::Mat & right_gray, double v, double ul,
                              double expected_disparity) const
{
    const double nan = std::numeric_limits<double>::quiet_NaN();
    // 以预测位置 (ul - disp, v) 为中心裁 ROI，横向留搜索余量
    const double cx_pred = ul - expected_disparity;
    const double s = p_.match_search_px;
    cv::Rect roi(static_cast<int>(cx_pred - s), static_cast<int>(v - s),
                 static_cast<int>(2 * s), static_cast<int>(2 * s));
    roi &= cv::Rect(0, 0, right_gray.cols, right_gray.rows);
    if (roi.width < 3 || roi.height < 3)
    {
        return nan;
    }

    const cv::Mat crop = right_gray(roi);
    cv::Mat bin;
    cv::threshold(crop, bin, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, {3, 3});
    cv::morphologyEx(bin, bin, cv::MORPH_OPEN, kernel);

    std::vector<std::vector<cv::Point>> cnts;
    cv::findContours(bin, cnts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    if (cnts.empty())
    {
        return nan;
    }

    // 取离预测 x 最近的亮斑（羽毛球只有一个，极线对齐后同一行）
    const auto best = std::min_element(
        cnts.begin(), cnts.end(), [&](const std::vector<cv::Point> & a,
                                      const std::vector<cv::Point> & b) {
            const cv::Moments ma = cv::moments(a);
            const cv::Moments mb = cv::moments(b);
            const double xa = (ma.m00 > 0) ? ma.m10 / ma.m00 + roi.x : cx_pred;
            const double xb = (mb.m00 > 0) ? mb.m10 / mb.m00 + roi.x : cx_pred;
            return std::fabs(xa - cx_pred) < std::fabs(xb - cx_pred);
        });

    const cv::Moments m = cv::moments(*best);
    if (m.m00 <= 0)
    {
        return nan;
    }
    return m.m10 / m.m00 + roi.x;  // 亚像素 x（全图坐标）
}

}  // namespace auto_aim
