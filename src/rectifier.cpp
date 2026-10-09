#include "rectifier.hpp"

#include <opencv2/imgproc.hpp>

namespace auto_aim
{

StereoRectifier::StereoRectifier(const std::string & maps_xml)
{
    cv::FileStorage fs(maps_xml, cv::FileStorage::READ);
    if (!fs.isOpened())
    {
        return;
    }
    fs["mapl1"] >> mapl1_;
    fs["mapl2"] >> mapl2_;
    fs["mapr1"] >> mapr1_;
    fs["mapr2"] >> mapr2_;
    ready_ = !mapl1_.empty() && !mapl2_.empty() && !mapr1_.empty() && !mapr2_.empty();
}

void StereoRectifier::rectify(const cv::Mat & L, const cv::Mat & R, cv::Mat & Lrect,
                              cv::Mat & Rrect) const
{
    if (!ready_)
    {
        return;
    }
    cv::remap(L, Lrect, mapl1_, mapl2_, cv::INTER_LINEAR);
    cv::remap(R, Rrect, mapr1_, mapr2_, cv::INTER_LINEAR);
}

}  // namespace auto_aim
