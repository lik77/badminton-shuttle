#pragma once
// 白羽毛球检测器（YOLO 候选框 → 框内经典精修）
//
// 数据流：
//   YOLO26-nano(单类 shuttle) ──> 候选框(全分辨率)
//        └─> refine() 框内: 阈值→开运算→最大轮廓→尺寸/圆度过滤→亚像素质心→等效直径
//        └─> ShuttleMeasurement{center, diameter_px, confidence}
//
// 候选框来源（按优先级）：
//   1) 已锁定 → EKF 预测点裁 ROI（最快）
//   2) 复用上一帧框（proposer_period 内）
//   3) YOLO 全帧提议
//   4) 背景减除(MOG2)+亮度 兜底（发射瞬间/小球 YOLO 抓不到时）

#include <opencv2/core.hpp>
#include <opencv2/video.hpp>
#include <cstdint>
#include <optional>
#include <string>

namespace auto_aim
{

struct ShuttleMeasurement
{
    bool valid{false};
    cv::Point2f center;      // 亚像素球心（全分辨率）
    float diameter_px{0.F};  // 等效直径(px)，尺寸测距用
    float confidence{0.F};   // 初值用圆度
};

// YOLO 候选框提议器接口。契约：propose() 返回的框必须是【全分辨率】坐标。
class ShuttleBoxProposer
{
  public:
    virtual ~ShuttleBoxProposer() = default;
    virtual cv::Rect propose(const cv::Mat & bgr_full, std::int64_t frame_id) = 0;
};

class ShuttleDetector
{
  public:
    ShuttleDetector(const std::string & config_path, ShuttleBoxProposer * proposer,
                    bool debug = false);

    // 主入口：一步完成「候选框 → 经典精修」。
    // predicted_px：EKF 预测球心（全分辨率），锁定后走 ROI；为空走 YOLO/复用/MOG2。
    ShuttleMeasurement detect(const cv::Mat & bgr_full, const cv::Mat & gray_full,
                              std::int64_t frame_id,
                              const std::optional<cv::Point2f> & predicted_px);

    // 经典精修（框内）。可独立调用。
    ShuttleMeasurement refine(const cv::Mat & gray_full, const cv::Rect & box) const;

  private:
    // 背景减除兜底：YOLO 失败时，全帧 MOG2 + 亮度找候选 blob。返回是否得到候选框。
    bool bg_fallback_box(const cv::Mat & gray_full, cv::Rect & box);

    bool debug_;

    double bright_thresh_;
    bool use_otsu_;
    int min_diameter_px_;
    int max_diameter_px_;
    double min_roundness_;
    int box_margin_;
    int proposer_period_;
    double motion_expand_;
    double max_jump_px_;

    ShuttleBoxProposer * proposer_;
    cv::Rect last_box_;
    std::int64_t last_propose_frame_{-1};
    bool have_last_box_{false};

    cv::Ptr<cv::BackgroundSubtractorMOG2> bg_sub_;  // 兜底候选源
};

}  // namespace auto_aim
