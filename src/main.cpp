// 白羽毛球感知主程序（完整 4 相机 / 两套双目）
//
// 数据流：
//   硬触发 → 8mm组(wide) + 16mm组(tele) 同步抓帧
//     ├─ 未锁定：wide 全帧粗定位
//     └─ 已锁定：按距离切换主组（远→tele 精轨迹，近→wide 高频），另一组兜底
//   测量 → 统一到世界系(+X前/+Y左/+Z上) → 一个 DragPredictor 融合 → 拦截点
//
// 鲁棒性：
//   - 连续丢 N 帧才解锁（短暂遮挡不重置）
//   - 丢帧仍照常 predict/intercept（预测穿透遮挡）
//   - 每组测量失败会依次 fallback：ROI→YOLO→背景减除
//
// 运行前改：configs/rigs.yaml（两组序列号/外参）+ kYoloOnnx。
#include <opencv2/opencv.hpp>

#include <cmath>
#include <cstdio>
#include <functional>
#include <optional>

#include "drag_predictor.hpp"
#include "rigs.hpp"
#include "shuttle_detector.hpp"
#include "yolo_proposer.hpp"

using namespace auto_aim;

const char * kYoloOnnx = "shuttle_yolo26.onnx";  // YOLO26 单类 ONNX

// 对一个双目组做完整测量：校正 → 左检测 → 右匹配 → 三角化 → 转世界系。
static bool measure(StereoRigUnit & unit, const StereoFrame & f, ShuttleDetector & det,
                    const DragPredictor & pred, bool locked,
                    const std::function<Eigen::Vector3d(const Eigen::Vector3d &)> & to_world,
                    const std::function<Eigen::Vector3d(const Eigen::Vector3d &)> & world_to_unit,
                    Eigen::Vector3d & out_p, float & out_dpx, cv::Point2f & out_center, float & out_conf)
{
    cv::Mat Lrect, Rrect;
    if (!unit.rectify(f, Lrect, Rrect)) return false;
    cv::Mat Lg, Rg;
    cv::cvtColor(Lrect, Lg, cv::COLOR_BGR2GRAY);
    cv::cvtColor(Rrect, Rg, cv::COLOR_BGR2GRAY);

    std::optional<cv::Point2f> pred_px;
    double Z_est = 8.0;
    if (locked)
    {
        const Eigen::Vector3d pu = world_to_unit(pred.state().head<3>());
        pred_px = unit.stereo().project(pu);
        Z_est = pu.z();
    }

    // 传真实帧号（修复：frame_id 写死 0 会导致 YOLO 只跑一次）
    auto mL = det.detect(Lrect, Lg, static_cast<std::int64_t>(f.frame_id), pred_px);
    if (!mL.valid) return false;

    out_center = mL.center;   // 检测球心（校正图坐标）
    out_conf = mL.confidence;  // 检测置信度（初值=圆度）

    const double disp = unit.stereo().params().f_px * unit.stereo().params().baseline / Z_est;
    const double ur = unit.stereo().match_right(Rg, mL.center.y, mL.center.x, disp);
    if (!std::isfinite(ur)) return false;

    Eigen::Vector3d pu;
    if (!unit.stereo().triangulate(mL.center.x, ur, mL.center.y, pu)) return false;
    out_p = to_world(pu);
    out_dpx = mL.diameter_px;
    return true;
}

int main()
{
    // 1) 两套双目
    RigSystem rigs;
    if (!rigs.load("configs/rigs.yaml"))
    {
        std::fprintf(stderr, "[main] 双组加载失败，检查 configs/rigs.yaml\n");
        return 1;
    }

    // 2) YOLO + 检测器（每组一个）
    YoloBoxProposer proposer("configs/yolo.yaml");  // 参数从 yaml 读（现场调无需重编译）
    if (!proposer.load(kYoloOnnx))
    {
        std::fprintf(stderr, "[main] YOLO 模型加载失败：%s\n", kYoloOnnx);
        return 1;
    }
    ShuttleDetector det_wide("configs/shuttle.yaml", &proposer, true);
    ShuttleDetector det_tele("configs/shuttle.yaml", &proposer, true);
    DragPredictor predictor("configs/drag_predictor.yaml");

    // 3) 主循环
    bool locked = false;
    double t_prev = -1.0;
    const int kMaxMissFrames = 10;  // 连续丢 N 帧才解锁
    int miss_count = 0;
    while (true)
    {
        RigFrames F;
        if (!rigs.grab(F)) continue;

        if (t_prev < 0.0) t_prev = F.wide.t_capture;
        const double dt = F.wide.t_capture - t_prev;
        t_prev = F.wide.t_capture;

        Eigen::Vector3d pw;
        float dpx = 0.0F;
        double f_px_used = 2319.0;
        bool measured = false;
        cv::Point2f det_center(-1.F, -1.F);
        int det_rig = -1;  // 0=wide 左，1=tele 左
        float det_conf = 0.0F;

        if (!locked)
        {
            // 未锁定：wide 全帧粗定位
            measured = measure(rigs.wide(), F.wide, det_wide, predictor, false,
                               [&](const Eigen::Vector3d & p) { return rigs.wide_to_world(p); },
                               [&](const Eigen::Vector3d & p) { return rigs.world_to_wide(p); },
                               pw, dpx, det_center, det_conf);
            if (measured)
            {
                det_rig = 0;
                predictor.reset(pw, F.wide.t_capture);
                locked = true;
                miss_count = 0;
                f_px_used = rigs.wide().stereo().params().f_px;
            }
        }
        else
        {
            // 已锁定：按距离切换主组
            const double dist = predictor.state().head<3>().norm();
            const bool far = dist > rigs.handoff_dist();
            if (far)
            {
                measured = measure(rigs.tele(), F.tele, det_tele, predictor, true,
                                   [&](const Eigen::Vector3d & p) { return rigs.tele_to_world(p); },
                                   [&](const Eigen::Vector3d & p) { return rigs.world_to_tele(p); },
                                   pw, dpx, det_center, det_conf);
                if (measured) { det_rig = 1; f_px_used = rigs.tele().stereo().params().f_px; }
                else
                {
                    measured = measure(rigs.wide(), F.wide, det_wide, predictor, true,
                                       [&](const Eigen::Vector3d & p) { return rigs.wide_to_world(p); },
                                       [&](const Eigen::Vector3d & p) { return rigs.world_to_wide(p); },
                                       pw, dpx, det_center, det_conf);
                    if (measured) { det_rig = 0; f_px_used = rigs.wide().stereo().params().f_px; }
                }
            }
            else
            {
                measured = measure(rigs.wide(), F.wide, det_wide, predictor, true,
                                   [&](const Eigen::Vector3d & p) { return rigs.wide_to_world(p); },
                                   [&](const Eigen::Vector3d & p) { return rigs.world_to_wide(p); },
                                   pw, dpx, det_center, det_conf);
                if (measured) { det_rig = 0; f_px_used = rigs.wide().stereo().params().f_px; }
                else
                {
                    measured = measure(rigs.tele(), F.tele, det_tele, predictor, true,
                                       [&](const Eigen::Vector3d & p) { return rigs.tele_to_world(p); },
                                       [&](const Eigen::Vector3d & p) { return rigs.world_to_tele(p); },
                                       pw, dpx, det_center, det_conf);
                    if (measured) { det_rig = 1; f_px_used = rigs.tele().stereo().params().f_px; }
                }
            }

            // 连续丢 N 帧才解锁（不 continue，下面照常 predict/intercept）
            if (measured) miss_count = 0;
            else if (++miss_count >= kMaxMissFrames) locked = false;
        }

        // 丢帧也照常推进 EKF + 预测拦截点（穿透遮挡）
        predictor.predict(dt);
        if (measured)
        {
            predictor.update_position(pw, F.wide.t_capture);
            predictor.update_size_px(dpx, f_px_used, F.wide.t_capture);
        }

        auto ic = predictor.intercept(0.9);
        if (ic.solvable)
        {
            std::printf("intercept t=%.3fs pos=(%.2f,%.2f,%.2f) conf=%.2f\n",
                        ic.t_hit, ic.pos.x(), ic.pos.y(), ic.pos.z(), ic.confidence);
        }

        // 可视化：2×2 四相机校正图
        cv::Mat wL, wR, tL, tR;
        rigs.wide().rectify(F.wide, wL, wR);
        rigs.tele().rectify(F.tele, tL, tR);

        // 缩放因子（校正图 1440×1080 → 显示 640×480）
        const float sx = 640.0F / wL.cols;
        const float sy = 480.0F / wL.rows;

        const cv::Size ds(640, 480);
        cv::resize(wL, wL, ds);
        cv::resize(wR, wR, ds);
        cv::resize(tL, tL, ds);
        cv::resize(tR, tR, ds);

        // 检测标记（画在缩放后的图上，尺寸可控、醒目）
        if (measured)
        {
            cv::Mat & cell = (det_rig == 1) ? tL : wL;
            const cv::Point c(static_cast<int>(det_center.x * sx), static_cast<int>(det_center.y * sy));
            const int rr = std::max(20, static_cast<int>(dpx * sx));
            cv::circle(cell, c, rr, cv::Scalar(0, 0, 255), 3);
            const int len = rr + 15;
            cv::line(cell, cv::Point(c.x - len, c.y), cv::Point(c.x + len, c.y), cv::Scalar(0, 0, 255), 2);
            cv::line(cell, cv::Point(c.x, c.y - len), cv::Point(c.x, c.y + len), cv::Scalar(0, 0, 255), 2);
            char info[64];
            std::snprintf(info, sizeof(info), "d=%.0f conf=%.2f", dpx, det_conf);
            cv::putText(cell, info, cv::Point(c.x + len, c.y - len), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
        }

        auto label = [](cv::Mat & m, const char * s) {
            cv::putText(m, s, {10, 25}, cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        };
        label(wL, "wide L");
        label(wR, "wide R");
        label(tL, "tele L");
        label(tR, "tele R");

        cv::Mat top, bot, grid;
        cv::hconcat(wL, wR, top);
        cv::hconcat(tL, tR, bot);
        cv::vconcat(top, bot, grid);

        char status[256];
        if (measured)
            std::snprintf(status, sizeof(status),
                          "locked=%d rig=%s 3D=(%.2f,%.2f,%.2f) dpx=%.0f",
                          locked ? 1 : 0, det_rig == 1 ? "tele" : "wide",
                          pw.x(), pw.y(), pw.z(), dpx);
        else
            std::snprintf(status, sizeof(status), "no detect  locked=%d", locked ? 1 : 0);
        cv::putText(grid, status, {10, grid.rows - 12}, cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(0, 255, 255), 2);

        cv::imshow("shuttle (2x2)", grid);
        int key = cv::waitKey(1);
        if (key == 'q' || key == 27) break;
    }
    return 0;
}
