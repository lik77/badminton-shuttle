#include "drag_predictor.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <utility>

namespace auto_aim
{

DragPredictor::DragPredictor(const std::string & config_path)
{
    auto yaml = YAML::LoadFile(config_path);
    g_ = yaml["g"].as<double>();
    sigma_accel_ = yaml["sigma_accel"].as<double>();
    sigma_k_ = yaml["sigma_k"].as<double>();
    sigma_pos_ = yaml["sigma_pos"].as<double>();
    sigma_range_ = yaml["sigma_range"].as<double>();
    k_init_ = yaml["k_init"].as<double>();
    nis_threshold_ = yaml["nis_threshold"].as<double>();
    hit_height_ = yaml["hit_height"].as<double>();
    min_updates_ = yaml["min_updates"].as<int>();
    max_ring_ = yaml["max_ring"].as<int>();

    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(7);
    x0[kK] = k_init_;
    Eigen::MatrixXd P0 = Eigen::MatrixXd::Zero(7, 7);
    P0.block<3, 3>(0, 0) = 1.0 * Eigen::Matrix3d::Identity();
    P0.block<3, 3>(3, 3) = 100.0 * Eigen::Matrix3d::Identity();
    P0(kK, kK) = 0.1 * 0.1;
    ekf_ = tools::ExtendedKalmanFilter(x0, P0);
}

void DragPredictor::reset(const Eigen::Vector3d & p0, double t0)
{
    ekf_.x.setZero();
    ekf_.x[kPx] = p0.x();
    ekf_.x[kPy] = p0.y();
    ekf_.x[kPz] = p0.z();
    ekf_.x[kK] = k_init_;
    ekf_.P.setZero();
    ekf_.P.block<3, 3>(0, 0) = 1.0 * Eigen::Matrix3d::Identity();
    ekf_.P.block<3, 3>(3, 3) = 100.0 * Eigen::Matrix3d::Identity();
    ekf_.P(kK, kK) = 0.1 * 0.1;
    t_now_ = t0;
    ring_.clear();
    push_snapshot();
    update_count_ = 0;
    converged_ = false;
}

void DragPredictor::predict(double dt)
{
    step(dt);
    t_now_ += dt;
    push_snapshot();
}

void DragPredictor::step(double dt)
{
    const double vx = ekf_.x[kVx], vy = ekf_.x[kVy], vz = ekf_.x[kVz], k = ekf_.x[kK];
    const double s = std::sqrt(vx * vx + vy * vy + vz * vz);
    const double ss = (s > 1e-9) ? s : 1e-9;

    auto f = [&](const Eigen::VectorXd & x) -> Eigen::VectorXd {
        Eigen::VectorXd y = x;
        const double vx_ = x[kVx], vy_ = x[kVy], vz_ = x[kVz], k_ = x[kK];
        const double s_ = std::sqrt(vx_ * vx_ + vy_ * vy_ + vz_ * vz_);
        y[kPx] += vx_ * dt;
        y[kPy] += vy_ * dt;
        y[kPz] += vz_ * dt;
        y[kVx] += -k_ * s_ * vx_ * dt;
        y[kVy] += -k_ * s_ * vy_ * dt;
        y[kVz] += (-g_ - k_ * s_ * vz_) * dt;
        return y;
    };

    Eigen::MatrixXd F = Eigen::MatrixXd::Identity(7, 7);
    F(kPx, kVx) = dt;
    F(kPy, kVy) = dt;
    F(kPz, kVz) = dt;

    Eigen::Matrix3d vv;
    vv << vx * vx, vx * vy, vx * vz,
          vy * vx, vy * vy, vy * vz,
          vz * vx, vz * vy, vz * vz;
    const Eigen::Matrix3d Jvv = -k * (ss * Eigen::Matrix3d::Identity() + vv / ss);
    F.block<3, 3>(kVx, kVx) += dt * Jvv;

    F(kVx, kK) = -ss * vx * dt;
    F(kVy, kK) = -ss * vy * dt;
    F(kVz, kK) = -ss * vz * dt;

    const double a = dt * dt * dt * dt / 4.0;
    const double b = dt * dt * dt / 2.0;
    const double c = dt * dt;
    const double va = sigma_accel_ * sigma_accel_;
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(7, 7);
    for (int i = 0; i < 3; ++i)
    {
        Q(kPx + i, kPx + i) += va * a;
        Q(kVx + i, kVx + i) += va * c;
        Q(kPx + i, kVx + i) += va * b;
        Q(kVx + i, kPx + i) += va * b;
    }
    Q(kK, kK) = sigma_k_ * sigma_k_ * dt;

    ekf_.predict(F, Q, f);
}

void DragPredictor::push_snapshot()
{
    Snapshot s;
    s.t = t_now_;
    s.x = ekf_.x;
    s.P = ekf_.P;
    ring_.push_back(std::move(s));
    if (static_cast<int>(ring_.size()) > max_ring_)
    {
        ring_.pop_front();
    }
}

bool DragPredictor::rewind_to(double t_capture)
{
    if (t_capture >= t_now_)
    {
        return true;  // 不是迟到观测，直接用当前状态
    }
    // 找最新一个 t <= t_capture 的快照
    auto it = std::upper_bound(ring_.begin(), ring_.end(), t_capture,
                               [](double val, const Snapshot & s) { return val < s.t; });
    if (it == ring_.begin())
    {
        return false;  // 观测太老，已滚出回放环
    }
    --it;
    ekf_.x = it->x;
    ekf_.P = it->P;
    step(t_capture - it->t);  // 推进到捕获时刻
    return true;
}

bool DragPredictor::update_position(const Eigen::Vector3d & p, double t_capture)
{
    const double orig_now = t_now_;
    if (!rewind_to(t_capture))
    {
        return false;  // 观测太老，丢弃
    }

    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(3, 7);
    H.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    const Eigen::Matrix3d R = sigma_pos_ * sigma_pos_ * Eigen::Matrix3d::Identity();

    const Eigen::Vector3d y = p - ekf_.x.head<3>();
    const Eigen::Matrix3d S = ekf_.P.block<3, 3>(0, 0) + R;
    last_nis_ = y.transpose() * S.inverse() * y;

    bool accepted = false;
    if (last_nis_ <= nis_threshold_)
    {
        ekf_.update(p, H, R);
        accepted = true;
        if (++update_count_ >= min_updates_)
        {
            converged_ = true;
        }
    }

    // 回放：从捕获时刻再预测回 now（更新已作用在正确历史时刻上）。
    if (t_capture < orig_now)
    {
        step(orig_now - t_capture);
    }
    t_now_ = orig_now;
    return accepted;
}

bool DragPredictor::update_size(double range, double t_capture)
{
    const double orig_now = t_now_;
    if (!rewind_to(t_capture))
    {
        return false;
    }

    const Eigen::Vector3d p = ekf_.x.head<3>();
    const double r = p.norm();
    if (r < 1e-6)
    {
        t_now_ = orig_now;
        return false;
    }

    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(1, 7);
    H.block<1, 3>(0, 0) = (p / r).transpose();
    Eigen::MatrixXd R(1, 1);
    R(0, 0) = sigma_range_ * sigma_range_;

    const double y = range - r;
    const double S = (H * ekf_.P * H.transpose())(0, 0) + R(0, 0);
    last_nis_ = y * y / S;

    bool accepted = false;
    if (last_nis_ <= nis_threshold_)
    {
        Eigen::VectorXd z(1);
        z(0) = range;
        auto h = [](const Eigen::VectorXd & x) {
            Eigen::VectorXd y(1);
            y(0) = x.head<3>().norm();
            return y;
        };
        ekf_.update(z, H, R, h);
        accepted = true;
    }

    if (t_capture < orig_now)
    {
        step(orig_now - t_capture);
    }
    t_now_ = orig_now;
    return accepted;
}

bool DragPredictor::update_size_px(double diameter_px, double f_px, double t_capture)
{
    if (diameter_px <= 1.0)
    {
        return false;
    }
    return update_size(0.066 * f_px / diameter_px, t_capture);
}

Intercept DragPredictor::intercept(double hit_height) const
{
    Intercept ic;
    const double dt = 0.005;
    const double max_t = 3.0;
    Eigen::VectorXd x = ekf_.x;
    double t = 0.0;

    for (int i = 0; i < static_cast<int>(max_t / dt); ++i)
    {
        const Eigen::Vector3d p_before = x.head<3>();
        const Eigen::Vector3d v_before = x.segment<3>(kVx);
        const double z_before = x[kPz];

        const double vx = x[kVx], vy = x[kVy], vz = x[kVz], k = x[kK];
        const double s = std::sqrt(vx * vx + vy * vy + vz * vz);
        x[kPx] += vx * dt;
        x[kPy] += vy * dt;
        x[kPz] += vz * dt;
        x[kVx] += -k * s * vx * dt;
        x[kVy] += -k * s * vy * dt;
        x[kVz] += (-g_ - k * s * vz) * dt;
        t += dt;

        if (x[kVz] < 0.0 && z_before >= hit_height && x[kPz] <= hit_height)
        {
            const double dz = z_before - x[kPz];
            const double frac = (dz > 1e-9) ? (z_before - hit_height) / dz : 0.0;
            ic.t_hit = t - dt * (1.0 - frac);
            ic.pos = p_before + frac * (x.head<3>() - p_before);
            ic.vel = v_before + frac * (x.segment<3>(kVx) - v_before);
            ic.confidence = 1.0 / (1.0 + ekf_.P.block<3, 3>(0, 0).trace());
            ic.solvable = true;
            return ic;
        }
    }
    return ic;
}

}  // namespace auto_aim
