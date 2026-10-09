#include "yolo_proposer.hpp"

#include <yaml-cpp/yaml.h>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>

namespace auto_aim
{

YoloBoxProposer::YoloBoxProposer() : p_{} {}

YoloBoxProposer::YoloBoxProposer(Params p) : p_(p) {}

YoloBoxProposer::YoloBoxProposer(const std::string & yaml_path) : p_{}
{
    try
    {
        const auto y = YAML::LoadFile(yaml_path);
        if (y["input_size"]) p_.input_size = y["input_size"].as<int>();
        if (y["conf_thresh"]) p_.conf_thresh = y["conf_thresh"].as<float>();
        if (y["nms_thresh"]) p_.nms_thresh = y["nms_thresh"].as<float>();
        if (y["class_id"]) p_.class_id = y["class_id"].as<int>();
        if (y["num_classes"]) p_.num_classes = y["num_classes"].as<int>();
    }
    catch (const std::exception & e)
    {
        std::fprintf(stderr, "[yolo] 读参数失败 %s: %s（用默认值）\n", yaml_path.c_str(), e.what());
    }
}

bool YoloBoxProposer::load(const std::string & onnx_path)
{
    try
    {
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "shuttle_yolo");
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(4);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_ = std::make_unique<Ort::Session>(*env_, onnx_path.c_str(), opts);
        loaded_ = true;
    }
    catch (const std::exception & e)
    {
        std::fprintf(stderr, "[yolo] 加载 ONNX 失败: %s\n", e.what());
        loaded_ = false;
    }
    return loaded_;
}

void YoloBoxProposer::letterbox(const cv::Mat & src, cv::Mat & dst, float & scale, int & padx,
                                int & pady) const
{
    scale = std::min(static_cast<float>(p_.input_size) / src.cols,
                     static_cast<float>(p_.input_size) / src.rows);
    const int new_w = std::max(1, static_cast<int>(std::round(src.cols * scale)));
    const int new_h = std::max(1, static_cast<int>(std::round(src.rows * scale)));
    cv::Mat resized;
    cv::resize(src, resized, {new_w, new_h});
    dst = cv::Mat(p_.input_size, p_.input_size, src.type(), cv::Scalar(114, 114, 114));
    padx = (p_.input_size - new_w) / 2;
    pady = (p_.input_size - new_h) / 2;
    resized.copyTo(dst(cv::Rect(padx, pady, new_w, new_h)));
}

cv::Rect YoloBoxProposer::remap(const Box & b, float scale, int padx, int pady,
                                const cv::Size & full) const
{
    const auto to_full = [&](float x, float y) {
        return cv::Point2f((x - padx) / scale, (y - pady) / scale);
    };
    const cv::Point2f tl = to_full(b.x1, b.y1);
    const cv::Point2f br = to_full(b.x2, b.y2);
    int x1 = std::max(0, static_cast<int>(tl.x));
    int y1 = std::max(0, static_cast<int>(tl.y));
    int x2 = std::min(full.width, static_cast<int>(std::ceil(br.x)));
    int y2 = std::min(full.height, static_cast<int>(std::ceil(br.y)));
    return cv::Rect(x1, y1, std::max(0, x2 - x1), std::max(0, y2 - y1));
}

std::vector<YoloBoxProposer::Box> YoloBoxProposer::nms(const std::vector<Box> & boxes) const
{
    std::vector<Box> kept;
    std::vector<int> idx(boxes.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&](int a, int b) { return boxes[a].conf > boxes[b].conf; });
    std::vector<bool> removed(boxes.size(), false);
    for (const int i : idx)
    {
        if (removed[i]) continue;
        kept.push_back(boxes[i]);
        for (const int j : idx)
        {
            if (removed[j] || i == j) continue;
            const float ix1 = std::max(boxes[i].x1, boxes[j].x1);
            const float iy1 = std::max(boxes[i].y1, boxes[j].y1);
            const float ix2 = std::min(boxes[i].x2, boxes[j].x2);
            const float iy2 = std::min(boxes[i].y2, boxes[j].y2);
            const float iw = std::max(0.F, ix2 - ix1);
            const float ih = std::max(0.F, iy2 - iy1);
            const float inter = iw * ih;
            const float a1 = (boxes[i].x2 - boxes[i].x1) * (boxes[i].y2 - boxes[i].y1);
            const float a2 = (boxes[j].x2 - boxes[j].x1) * (boxes[j].y2 - boxes[j].y1);
            const float uni = a1 + a2 - inter;
            if (uni > 0 && inter / uni > p_.nms_thresh) removed[j] = true;
        }
    }
    return kept;
}

std::vector<YoloBoxProposer::Box> YoloBoxProposer::infer(const cv::Mat & bgr_full)
{
    std::vector<Box> out;
    if (!loaded_) return out;

    // 1) letterbox
    cv::Mat letter;
    float scale;
    int padx, pady;
    letterbox(bgr_full, letter, scale, padx, pady);

    // 2) 预处理：BGR->RGB、1/255、NCHW（等价 blobFromImage(swapRB=true)）
    cv::Mat blob = cv::dnn::blobFromImage(letter, 1.0 / 255.0,
                                          cv::Size(p_.input_size, p_.input_size),
                                          cv::Scalar(0, 0, 0), true, false);

    // 3) 输入张量
    std::vector<std::int64_t> in_shape{1, 3, p_.input_size, p_.input_size};
    const std::size_t numel = static_cast<std::size_t>(1) * 3 * p_.input_size * p_.input_size;
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value in_tensor = Ort::Value::CreateTensor<float>(
        mem, reinterpret_cast<float *>(blob.data), numel, in_shape.data(), in_shape.size());

    // 4) 前向
    const char * const in_names[] = {"images"};
    const char * const out_names[] = {"output0"};
    Ort::RunOptions run_opts;
    auto outs = session_->Run(run_opts, in_names, &in_tensor, 1, out_names, 1);

    // 5) 解码：Ultralytics 导出布局 [1, 4+nc, N]（每列 = cx,cy,w,h,class0..classN）
    const float * data = outs[0].GetTensorData<float>();
    const auto shape = outs[0].GetTensorTypeAndShapeInfo().GetShape();
    if (shape.size() < 3) return out;
    const int C = static_cast<int>(shape[1]);
    const int N = static_cast<int>(shape[2]);
    if (C < 5) return out;

    for (int i = 0; i < N; ++i)
    {
        const float conf = data[(4 + p_.class_id) * N + i];
        if (conf < p_.conf_thresh) continue;
        const float cx = data[0 * N + i], cy = data[1 * N + i];
        const float w = data[2 * N + i], h = data[3 * N + i];
        Box b;
        b.x1 = cx - w * 0.5F;
        b.y1 = cy - h * 0.5F;
        b.x2 = cx + w * 0.5F;
        b.y2 = cy + h * 0.5F;
        b.conf = conf;
        b.cls = p_.class_id;
        cv::Rect r = remap(b, scale, padx, pady, bgr_full.size());
        if (r.width <= 0 || r.height <= 0) continue;
        b.x1 = r.x;
        b.y1 = r.y;
        b.x2 = r.x + r.width;
        b.y2 = r.y + r.height;
        out.push_back(b);
    }
    return out;
}

cv::Rect YoloBoxProposer::propose(const cv::Mat & bgr_full, std::int64_t frame_id)
{
    (void)frame_id;
    if (!loaded_) return {};
    auto boxes = nms(infer(bgr_full));
    if (boxes.empty()) return {};
    const auto best = std::max_element(
        boxes.begin(), boxes.end(),
        [](const Box & a, const Box & b) { return a.conf < b.conf; });
    return cv::Rect(static_cast<int>(best->x1), static_cast<int>(best->y1),
                    static_cast<int>(best->x2 - best->x1), static_cast<int>(best->y2 - best->y1));
}

}  // namespace auto_aim
