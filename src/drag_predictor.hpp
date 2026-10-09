#pragma once
// 羽毛球阻力模型 EKF 预测器（DragPredictor）——含「曝光时刻时间戳 + 延迟观测回放」
//
// 设计来源：
//   1) 阻力模型 EKF：复用 tools/extended_kalman_filter（NIS/NEES 门控泛型 EKF），
//      过程模型换成羽毛球二次空气阻力 + 重力，状态加一维 k 在线估计。
//   2) 延迟观测回放：照 CUAS/cuas-fw/control/eskf.cpp 的「环形快照 + observe(z, t_capture)」——
//      视觉观测（检测+三角化）有几十 ms 延迟，观测到达时它已经是"过去"的状态；
//      这里把状态回卷到捕获时刻、更新、再预测回 now，避免用过期状态做校正。
//   3) 曝光时刻时间戳：观测携带 t_capture（相机设备时间戳，照 CUAS FrameSource.h 的
//      hwCorrectedStamp 校准到单调时钟），而不是"处理完成的时刻"。
//
// 状态 x = [px,py,pz, vx,vy,vz, k]，+Z 向上，重力只作用在 -Z，k 为阻力系数(终速6.7→0.22)。
//
// 用法（主循环）：
//   predictor.predict(dt);                          // 每个节拍推进
//   predictor.update_position(p3, t_capture);       // 双目 3D（带捕获时刻）
//   predictor.update_size_px(d_px, f_px, t_capture);// 尺寸测距
//   predictor.intercept(0.9);                       // 拦截点

#include <Eigen/Dense>
#include <cstddef>
#include <deque>
#include <string>

#include "tools/extended_kalman_filter.hpp"

namespace auto_aim
{

struct Intercept
{
    bool solvable{false};
    double t_hit{0.0};
    Eigen::Vector3d pos{0, 0, 0};
    Eigen::Vector3d vel{0, 0, 0};
    double confidence{0.0};
};

class DragPredictor
{
  public:
    explicit DragPredictor(const std::string & config_path);

    // 重新初始化。t0 为 t_capture 所用时钟的起点（默认 0，秒）。
    void reset(const Eigen::Vector3d & p0, double t0 = 0.0);

    // 推进一个节拍（dt 秒），并快照当前状态进回放环。
    void predict(double dt);

    // 3D 位置观测（双目三角化，已变换到世界/身体系），t_capture=曝光时刻。
    bool update_position(const Eigen::Vector3d & p, double t_capture);

    // 尺寸测距观测，range = 0.066*f_px/d_px 的便捷封装。
    bool update_size_px(double diameter_px, double f_px, double t_capture);

    // 原始距离观测（h(x)=|p|）。
    bool update_size(double range, double t_capture);

    Intercept intercept(double hit_height) const;

    Eigen::VectorXd state() const { return ekf_.x; }
    double now() const { return t_now_; }
    bool converged() const { return converged_; }
    double last_nis() const { return last_nis_; }

  private:
    static constexpr int kPx = 0, kPy = 1, kPz = 2, kVx = 3, kVy = 4, kVz = 5, kK = 6;

    struct Snapshot
    {
        double t{0.0};
        Eigen::VectorXd x;
        Eigen::MatrixXd P;
    };

    // 纯推进：只把 ekf_.x/P 沿阻力模型推进 dt，不动 t_now_ / 回放环。
    void step(double dt);
    void push_snapshot();
    // 把状态回卷到 t_capture（返回 false=观测太老，已滚出回放环）。
    bool rewind_to(double t_capture);

    tools::ExtendedKalmanFilter ekf_;
    std::deque<Snapshot> ring_;
    double t_now_{0.0};

    double g_;
    double sigma_accel_;
    double sigma_k_;
    double sigma_pos_;
    double sigma_range_;
    double k_init_;
    double nis_threshold_;
    double hit_height_;
    int min_updates_;
    int max_ring_;

    int update_count_{0};
    bool converged_{false};
    double last_nis_{0.0};
};

}  // namespace auto_aim
