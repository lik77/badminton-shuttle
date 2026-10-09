#include "shuttle_detector.hpp"

#include <opencv2/imgproc.hpp>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace auto_aim
{

namespace
{
inline cv::Rect clamp_to_image(const cv::Rect & r, const cv::Size & sz)
{
    return r & cv::Rect(0, 0, sz.width, sz.height);
}
}  // namespace

ShuttleDetector::ShuttleDetector(const std::string & config_path, ShuttleBoxProposer * proposer,
                                 bool debug)
: debug_(debug), proposer_(proposer)
{
    auto yaml = YAML::LoadFile(config_path);
    bright_thresh_ = yaml["bright_thresh"].as<double>();
    use_otsu_ = yaml["use_otsu"].as<bool>();
    min_diameter_px_ = yaml["min_diameter_px"].as<int>();
    max_diameter_px_ = yaml["max_diameter_px"].as<int>();
    min_roundness_ = yaml["min_roundness"].as<double>();
    box_margin_ = yaml["box_margin"].as<int>();
    proposer_period_ = yaml["proposer_period"].as<int>();
    motion_expand_ = yaml["motion_expand"].as<double>();
    max_jump_px_ = yaml["max_jump_px"].as<double>();

    // 背景减除兜底（发射瞬间/小球 YOLO 抓不到时用）
    bg_sub_ = cv::createBackgroundSubtractorMOG2(500, 16, false);
}

ShuttleMeasurement ShuttleDetector::detect(const cv::Mat & bgr_full, const cv::Mat & gray_full,
                                           std::int64_t frame_id,
                                           const std::optional<cv::Point2f> & predicted_px)
{
    cv::Rect box;
    bool have_box = false;
    bool from_roi = false;

    if (predicted_px)
    {
        // 已锁定：EKF 预测点 ROI，不再调 YOLO（省算力、低延迟）
        const int half = box_margin_ + static_cast<int>(motion_expand_);
        box = cv::Rect(static_cast<int>(predicted_px->x) - half,
                       static_cast<int>(predicted_px->y) - half, 2 * half, 2 * half);
        have_box = true;
        from_roi = true;
    }
    else if (have_last_box_ && (frame_id - last_propose_frame_) < proposer_period_)
    {
        // 复用上一帧 YOLO 框，按最大运动外扩
        box = last_box_;
        const int e = static_cast<int>(motion_expand_);
        box.x -= e; box.y -= e; box.width += 2 * e; box.height += 2 * e;
        have_box = true;
    }
    else if (proposer_)
    {
        // 未锁定 / 到期：调 YOLO 出候选框
        box = proposer_->propose(bgr_full, frame_id);
        if (box.width > 0 && box.height > 0)
        {
            last_box_ = box;
            last_propose_frame_ = frame_id;
            have_last_box_ = true;
            have_box = true;
        }
        else
        {
            have_last_box_ = false;
        }
    }

    // P1-3：YOLO 没出框时，背景减除 + 亮度兜底（发射瞬间/小球抓不到）
    if (!have_box && bg_sub_)
    {
        have_box = bg_fallback_box(gray_full, box);
    }

    if (!have_box)
    {
        return {};
    }

    box = clamp_to_image(box, gray_full.size());
    ShuttleMeasurement m = refine(gray_full, box);

    // P1-4：锁定后 ROI 精修失败 → 同帧退回 YOLO 全帧（别等 N 帧才重捕获）
    if (!m.valid && from_roi && proposer_)
    {
        box = proposer_->propose(bgr_full, frame_id);
        if (box.width > 0 && box.height > 0)
        {
            last_box_ = box;
            last_propose_frame_ = frame_id;
            have_last_box_ = true;
            box = clamp_to_image(box, gray_full.size());
            m = refine(gray_full, box);
        }
    }

    // 锁定后的空间校验：离预测点太远判为噪声
    if (m.valid && predicted_px)
    {
        if (cv::norm(m.center - *predicted_px) > max_jump_px_)
        {
            m.valid = false;
        }
    }
    return m;
}

bool ShuttleDetector::bg_fallback_box(const cv::Mat & gray_full, cv::Rect & box)
{
    // 背景减除 → 前景掩膜
    cv::Mat fg;
    bg_sub_->apply(gray_full, fg);

    // 亮度掩膜（白球是亮目标）
    cv::Mat bright;
    cv::threshold(gray_full, bright, bright_thresh_, 255, cv::THRESH_BINARY);

    // 前景 ∩ 亮度
    cv::Mat mask;
    cv::bitwise_and(fg, bright, mask);
    const cv::Mat k = cv::getStructuringElement(cv::MORPH_ELLIPSE, {3, 3});
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, k);

    std::vector<std::vector<cv::Point>> cnts;
    cv::findContours(mask, cnts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (cnts.empty())
    {
        return false;
    }
    const auto best = std::max_element(
        cnts.begin(), cnts.end(),
        [](const std::vector<cv::Point> & a, const std::vector<cv::Point> & b) {
            return cv::contourArea(a) < cv::contourArea(b);
        });

    // 取外接框，外扩 margin 给 refine 留余量
    cv::Rect r = cv::boundingRect(*best);
    r.x -= box_margin_;
    r.y -= box_margin_;
    r.width += 2 * box_margin_;
    r.height += 2 * box_margin_;
    box = r;
    return true;
}

ShuttleMeasurement ShuttleDetector::refine(const cv::Mat & gray_full, const cv::Rect & box) const
{
    ShuttleMeasurement m;
    if (box.width < 3 || box.height < 3)
    {
        return m;
    }

    cv::Rect r = box;
    r.x -= box_margin_;
    r.y -= box_margin_;
    r.width += 2 * box_margin_;
    r.height += 2 * box_margin_;
    r = clamp_to_image(r, gray_full.size());
    if (r.width < 3 || r.height < 3)
    {
        return m;
    }
    const cv::Mat crop = gray_full(r);

    // 阈值：大 ROI 用 Otsu（自适应、抗工频闪烁）；小 ROI（锁定后远场小球）用固定阈值，
    // 因为 Otsu 在几十像素小图上双峰假设不成立、阈值不稳（P2-5 修复）。
    const bool large_crop = (crop.cols >= 256 && crop.rows >= 256);
    cv::Mat bin;
    if (use_otsu_ && large_crop)
    {
        cv::threshold(crop, bin, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    }
    else
    {
        cv::threshold(crop, bin, bright_thresh_, 255, cv::THRESH_BINARY);
    }

    const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, {3, 3});
    cv::morphologyEx(bin, bin, cv::MORPH_OPEN, kernel);// 开运算去噪

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bin, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    if (contours.empty())
    {
        return m;
    }
    // 挑面积最大的轮廓（整条球）；质心用亮度加权（自然收敛到最亮的球头，不假设形状）
    int best_idx = 0;
    double best_area = -1.0;
    for (std::size_t i = 0; i < contours.size(); ++i)
    {
        const double a = cv::contourArea(contours[i]);
        if (a > best_area)
        {
            best_area = a;
            best_idx = static_cast<int>(i);
        }
    }

    const double area = cv::contourArea(contours[best_idx]);
    const double per = cv::arcLength(contours[best_idx], true);
    if (per <= 0.0)
    {
        return m;
    }
    const double roundness = 4.0 * CV_PI * area / (per * per);
    const double dpx = 2.0 * std::sqrt(area / CV_PI);
    if (dpx < min_diameter_px_ || dpx > max_diameter_px_)
    {
        return m;
    }
    if (roundness < min_roundness_)
    {
        return m;
    }

    cv::Mat mask = cv::Mat::zeros(crop.size(), CV_8U);
    cv::drawContours(mask, contours, best_idx, 255, cv::FILLED);

    double sx = 0.0, sy = 0.0, sw = 0.0;
    for (int y = 0; y < crop.rows; ++y)
    {
        const uchar * gp = crop.ptr<uchar>(y);
        const uchar * mp = mask.ptr<uchar>(y);
        for (int x = 0; x < crop.cols; ++x)
        {
            if (mp[x])
            {
                const double w = gp[x];
                sx += w * x;
                sy += w * y;
                sw += w;
            }
        }
    }
    if (sw <= 0.0)
    {
        return m;
    }

    m.center = cv::Point2f(static_cast<float>(r.x + sx / sw), static_cast<float>(r.y + sy / sw));
    m.diameter_px = static_cast<float>(dpx);
    m.confidence = static_cast<float>(roundness);
    m.valid = true;

    if (debug_)
    {
        std::fprintf(stderr, "[ShuttleDetector] center=(%.2f,%.2f) dpx=%.1f round=%.2f\n",
                     m.center.x, m.center.y, m.diameter_px, roundness);
    }
    return m;
}

}  // namespace auto_aim
