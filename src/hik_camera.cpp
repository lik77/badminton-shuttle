#include "hik_camera.hpp"

#include <MvCameraControl.h>
#include <opencv2/imgproc.hpp>

#include <cstdio>
#include <cstring>
#include <unordered_map>

namespace auto_aim
{

namespace
{
// 把 MVS 的定长 char 缓冲转成 std::string（序列号是 chSerialNumber[...]）
std::string bounded(const unsigned char * buf, std::size_t cap)
{
    std::size_t n = 0;
    while (n < cap && buf[n] != 0) ++n;
    return std::string(reinterpret_cast<const char *>(buf), n);
}

std::string usb_serial(const MV_CC_DEVICE_INFO * info)
{
    if (info->nTLayerType == MV_USB_DEVICE)
    {
        return bounded(info->SpecialInfo.stUsb3VInfo.chSerialNumber,
                       sizeof(info->SpecialInfo.stUsb3VInfo.chSerialNumber));
    }
    return {};
}
}  // namespace

HikCamera::~HikCamera() { close(); }

bool HikCamera::open(const std::string & serial, double exposure_us, double gain, int fps,
                     bool external_trigger)
{
    MV_CC_DEVICE_INFO_LIST list;
    std::memset(&list, 0, sizeof(list));
    unsigned int ret = MV_CC_EnumDevices(MV_USB_DEVICE, &list);
    if (ret != MV_OK || list.nDeviceNum == 0)
    {
        std::fprintf(stderr, "[hik] 枚举相机失败/无相机 ret=0x%x\n", ret);
        return false;
    }

    MV_CC_DEVICE_INFO * sel = nullptr;
    for (unsigned int i = 0; i < list.nDeviceNum; ++i)
    {
        MV_CC_DEVICE_INFO * info = list.pDeviceInfo[i];
        if (serial.empty() || usb_serial(info) == serial)
        {
            sel = info;
            break;
        }
    }
    if (sel == nullptr)
    {
        std::fprintf(stderr, "[hik] 未找到序列号 %s\n", serial.c_str());
        return false;
    }

    ret = MV_CC_CreateHandle(&handle_, sel);
    if (ret != MV_OK) return false;
    ret = MV_CC_OpenDevice(handle_);
    if (ret != MV_OK) { close(); return false; }

    // 固定配置（白平衡自动、曝光/增益手动）
    set_enum("BalanceWhiteAuto", MV_BALANCEWHITE_AUTO_CONTINUOUS);
    set_enum("ExposureAuto", MV_EXPOSURE_AUTO_MODE_OFF);
    set_enum("GainAuto", MV_GAIN_MODE_OFF);
    set_float("ExposureTime", exposure_us);
    set_float("Gain", gain);

    if (external_trigger)
    {
        // 硬触发：外部 Line0 给触发信号（MCU 主时钟），四台相机同步曝光
        set_enum("TriggerMode", MV_TRIGGER_MODE_ON);
        set_enum("TriggerSource", MV_TRIGGER_SOURCE_LINE0);
    }
    else
    {
        set_enum("TriggerMode", MV_TRIGGER_MODE_OFF);
        MV_CC_SetFrameRate(handle_, fps);
    }

    ret = MV_CC_StartGrabbing(handle_);
    if (ret != MV_OK) { close(); return false; }
    return true;
}

void HikCamera::close()
{
    if (handle_ == nullptr) return;
    MV_CC_StopGrabbing(handle_);
    MV_CC_CloseDevice(handle_);
    MV_CC_DestroyHandle(handle_);
    handle_ = nullptr;
}

bool HikCamera::grab(cv::Mat & bgr, std::uint64_t & dev_ts_ns, std::uint32_t & frame_num,
                     int timeout_ms)
{
    MV_FRAME_OUT raw;
    std::memset(&raw, 0, sizeof(raw));
    unsigned int ret = MV_CC_GetImageBuffer(handle_, &raw, timeout_ms);
    if (ret != MV_OK)
    {
        std::fprintf(stderr, "[hik] GetImageBuffer 失败/超时 ret=0x%x\n", ret);
        return false;
    }

    // 原始 Bayer 单通道图
    cv::Mat raw8(cv::Size(raw.stFrameInfo.nWidth, raw.stFrameInfo.nHeight), CV_8U, raw.pBufAddr);

    // Bayer -> RGB（相机输出 Bayer8；用 OpenCV 转，比 MV_CC_ConvertPixelType 简单）
    // 注意：IMX273 实际 Bayer 模式与 SDK 上报的红蓝相反，*2RGB 解出来的"RGB"
    //       其实已经是正确 BGR；再转 RGB2BGR 会红蓝互换（红色变蓝），所以不要再转。
    static const std::unordered_map<MvGvspPixelType, cv::ColorConversionCodes> kMap = {
        {PixelType_Gvsp_BayerGR8, cv::COLOR_BayerGR2RGB},
        {PixelType_Gvsp_BayerRG8, cv::COLOR_BayerRG2RGB},
        {PixelType_Gvsp_BayerGB8, cv::COLOR_BayerGB2RGB},
        {PixelType_Gvsp_BayerBG8, cv::COLOR_BayerBG2RGB},  // Mono8 直接给灰度
    };
    auto it = kMap.find(raw.stFrameInfo.enPixelType);
    if (it != kMap.end())
    {
        cv::cvtColor(raw8, bgr, it->second);
    }
    else
    {
        bgr = raw8;  // 已经是 Mono8 之类的，直接给灰度（调用方按需处理）
    }

    // 设备时间戳：高32位 | 低32位，单位 ns（1 tick = 1ns）
    dev_ts_ns = (static_cast<std::uint64_t>(raw.stFrameInfo.nDevTimeStampHigh) << 32) |
                static_cast<std::uint64_t>(raw.stFrameInfo.nDevTimeStampLow);
    frame_num = raw.stFrameInfo.nFrameNum;  // 帧号，单调递增，用于检测器复用框判断

    MV_CC_FreeImageBuffer(handle_, &raw);
    return true;
}

void HikCamera::set_float(const char * name, double value)
{
    if (MV_CC_SetFloatValue(handle_, name, static_cast<float>(value)) != MV_OK)
        std::fprintf(stderr, "[hik] set %s 失败\n", name);
}

void HikCamera::set_enum(const char * name, unsigned int value)
{
    if (MV_CC_SetEnumValue(handle_, name, value) != MV_OK)
        std::fprintf(stderr, "[hik] set %s 失败\n", name);
}

}  // namespace auto_aim
