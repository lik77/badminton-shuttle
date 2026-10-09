#include "rigs.hpp"

#include <yaml-cpp/yaml.h>

#include <cstdio>

namespace auto_aim
{

namespace
{
Eigen::Matrix3d load_mat3(const YAML::Node & n)
{
    Eigen::Matrix3d m;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            m(r, c) = n[r * 3 + c].as<double>();
    return m;
}
Eigen::Vector3d load_vec3(const YAML::Node & n)
{
    return Eigen::Vector3d(n[0].as<double>(), n[1].as<double>(), n[2].as<double>());
}
RigConfig load_rig(const YAML::Node & n)
{
    RigConfig c;
    c.name = n["name"].as<std::string>();
    c.left_serial = n["left_serial"].as<std::string>();
    c.right_serial = n["right_serial"].as<std::string>();
    c.exposure_us = n["exposure_us"].as<double>();
    c.gain = n["gain"].as<double>();
    c.fps = n["fps"].as<int>();
    c.external_trigger = n["external_trigger"].as<bool>();
    c.stereo_yaml = n["stereo_yaml"].as<std::string>();
    c.rectifier_xml = n["rectifier_xml"].as<std::string>();
    return c;
}
}  // namespace

bool StereoRigUnit::open(const RigConfig & cfg)
{
    if (!cam_.open(cfg.left_serial, cfg.right_serial, cfg.exposure_us, cfg.gain, cfg.fps,
                   cfg.external_trigger))
    {
        std::fprintf(stderr, "[rig] 组 %s 相机打开失败\n", cfg.name.c_str());
        return false;
    }
    rect_ = std::make_unique<StereoRectifier>(cfg.rectifier_xml);
    rig_ = std::make_unique<StereoRig>(cfg.stereo_yaml);
    if (!rect_->ready())
        std::fprintf(stderr, "[rig] 组 %s 校正映射未加载：%s\n", cfg.name.c_str(),
                     cfg.rectifier_xml.c_str());
    return true;
}

void StereoRigUnit::close() { cam_.close(); }

bool StereoRigUnit::rectify(const StereoFrame & f, cv::Mat & L, cv::Mat & R) const
{
    if (!rect_ || !rect_->ready()) return false;
    rect_->rectify(f.left, f.right, L, R);
    return !L.empty() && !R.empty();
}

bool RigSystem::load(const std::string & rigs_yaml)
{
    auto y = YAML::LoadFile(rigs_yaml);
    R_c2w_ = load_mat3(y["R_c2w"]);
    R_t2w_ = load_mat3(y["R_t2w"]);
    t_t2w_ = load_vec3(y["t_t2w"]);
    handoff_dist_ = y["handoff_dist"].as<double>();

    if (!wide_.open(load_rig(y["wide"]))) return false;
    if (!tele_.open(load_rig(y["tele"]))) return false;
    return wide_.ready() && tele_.ready();
}

void RigSystem::close()
{
    wide_.close();
    tele_.close();
}

bool RigSystem::grab(RigFrames & f, int timeout_ms)
{
    if (!wide_.grab(f.wide, timeout_ms)) return false;
    if (!tele_.grab(f.tele, timeout_ms)) return false;
    return true;
}

Eigen::Vector3d RigSystem::wide_to_world(const Eigen::Vector3d & p) const { return R_c2w_ * p; }

Eigen::Vector3d RigSystem::tele_to_world(const Eigen::Vector3d & p) const
{
    return R_c2w_ * (R_t2w_ * p + t_t2w_);
}

Eigen::Vector3d RigSystem::world_to_wide(const Eigen::Vector3d & p) const
{
    return R_c2w_.transpose() * p;
}

Eigen::Vector3d RigSystem::world_to_tele(const Eigen::Vector3d & p) const
{
    return R_t2w_.transpose() * (R_c2w_.transpose() * p - t_t2w_);
}

}  // namespace auto_aim
