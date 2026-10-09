#include "stereo_camera.hpp"

namespace auto_aim
{

bool StereoCamera::open(const std::string & left_serial, const std::string & right_serial,
                        double exposure_us, double gain, int fps, bool external_trigger)
{
    bool ok_l = left_.open(left_serial, exposure_us, gain, fps, external_trigger);
    bool ok_r = right_.open(right_serial, exposure_us, gain, fps, external_trigger);
    if (!ok_l || !ok_r)
    {
        close();
        return false;
    }
    return true;
}

void StereoCamera::close()
{
    left_.close();
    right_.close();
}

bool StereoCamera::grab(StereoFrame & f, int timeout_ms)
{
    std::uint64_t ts_l = 0, ts_r = 0;
    std::uint32_t fn_l = 0, fn_r = 0;
    if (!left_.grab(f.left, ts_l, fn_l, timeout_ms)) return false;
    if (!right_.grab(f.right, ts_r, fn_r, timeout_ms)) return false;
    f.t_capture = static_cast<double>(ts_l) * 1e-9;  // ns -> 秒（左相机时基）
    f.frame_id = fn_l;  // 帧号，喂给检测器
    (void)ts_r; (void)fn_r;
    return true;
}

bool StereoCamera::isOpen() const { return left_.isOpen() && right_.isOpen(); }

}  // namespace auto_aim
