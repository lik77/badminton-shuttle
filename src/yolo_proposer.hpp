#pragma once
// YOLO26 候选框提议器（ONNX Runtime 推理实现）。
// letterbox、推理、解码、remap 回全分辨率、NMS、单类取最高分。
#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "shuttle_detector.hpp"

namespace auto_aim
{

class YoloBoxProposer : public ShuttleBoxProposer
{
  public:
    struct Params
    {
        int input_size{1280};       // 8m 处球仅 19–38px，用高分辨率输入
        float conf_thresh{0.35F};
        float nms_thresh{0.45F};
        int class_id{0};            // "shuttle" 类别 id（单类=0）
        int num_classes{1};         // 类别数（单类=1）
    };

    YoloBoxProposer();
    explicit YoloBoxProposer(Params p);
    explicit YoloBoxProposer(const std::string & yaml_path);  // 从 yaml 读参数
    ~YoloBoxProposer() override = default;

    bool load(const std::string & onnx_path);  // 加载 ONNX 模型
    cv::Rect propose(const cv::Mat & bgr_full, std::int64_t frame_id) override;

  private:
    struct Box
    {
        float x1, y1, x2, y2, conf;
        int cls;
    };

    std::vector<Box> infer(const cv::Mat & bgr_full);  // letterbox + 推理 + 解码(全分辨率)
    void letterbox(const cv::Mat & src, cv::Mat & dst, float & scale, int & padx, int & pady) const;
    std::vector<Box> nms(const std::vector<Box> & boxes) const;
    cv::Rect remap(const Box & b, float scale, int padx, int pady, const cv::Size & full) const;

    Params p_;
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    bool loaded_{false};
};

}  // namespace auto_aim
