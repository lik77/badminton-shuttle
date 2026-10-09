// 双目标定采图工具 v3（独立文件，不改 shuttle/ 现有代码）
// 棋盘格：11×8 内角点（12×9 格子）。检测横竖两个方向都试，自动/手动存图。
// 用法:
//   ./capture_stereo                      自动取前两台（左=第1台，右=第2台）
//   ./capture_stereo <左序列号> <右序列号>   指定左右
// 按键:
//   [a] 切换 自动/手动（默认自动）
//   [s]/[空格] 手动存一对
//   [d] 删除最后一对
//   [q]/[Esc] 退出
#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <MvCameraControl.h>
#include "hik_camera.hpp"

using namespace auto_aim;

static const float kPi = 3.14159265f;

// 检测棋盘格：横竖两个方向都试（11×8 或 8×11），先 FAST_CHECK 快速筛、失败再完整检测。
static bool detect_board(const cv::Mat & gray, std::vector<cv::Point2f> & corners, cv::Size & used)
{
    const cv::Size sizes[2] = { cv::Size(11, 8), cv::Size(8, 11) };
    for (int i = 0; i < 2; ++i)
    {
        std::vector<cv::Point2f> c;
        if (cv::findChessboardCorners(gray, sizes[i], c, cv::CALIB_CB_FAST_CHECK))
        {
            corners = c; used = sizes[i]; return true;
        }
    }
    return false;
}

static std::string bounded(const unsigned char * buf, std::size_t cap)
{
    std::size_t n = 0;
    while (n < cap && buf[n] != 0) ++n;
    return std::string(reinterpret_cast<const char *>(buf), n);
}

static std::string usb_serial(const MV_CC_DEVICE_INFO * info)
{
    if (info->nTLayerType == MV_USB_DEVICE)
        return bounded(info->SpecialInfo.stUsb3VInfo.chSerialNumber,
                       sizeof(info->SpecialInfo.stUsb3VInfo.chSerialNumber));
    return {};
}

static std::vector<std::string> enum_serials()
{
    MV_CC_DEVICE_INFO_LIST list;
    std::memset(&list, 0, sizeof(list));
    std::vector<std::string> out;
    if (MV_CC_EnumDevices(MV_USB_DEVICE, &list) != MV_OK) return out;
    for (unsigned int i = 0; i < list.nDeviceNum; ++i)
        out.push_back(usb_serial(list.pDeviceInfo[i]));
    return out;
}

static int next_index()
{
    mkdir("left", 0777);
    mkdir("right", 0777);
    int mx = -1;
    DIR * d = opendir("left");
    if (d)
    {
        struct dirent * e;
        while ((e = readdir(d)) != nullptr)
        {
            int n = 0;
            if (std::sscanf(e->d_name, "%d", &n) == 1 && n > mx) mx = n;
        }
        closedir(d);
    }
    return mx + 1;
}

static float angle_diff(float a, float b)
{
    float d = a - b;
    while (d > kPi)  d -= 2 * kPi;
    while (d < -kPi) d += 2 * kPi;
    return d;
}

int main(int argc, char ** argv)
{
    std::vector<std::string> serials = enum_serials();
    printf("检测到 %zu 台相机:\n", serials.size());
    for (std::size_t i = 0; i < serials.size(); ++i)
        printf("  [%zu] %s\n", i, serials[i].c_str());

    if (serials.size() < 2)
    {
        std::fprintf(stderr, "双目标定需要【同组左右两台】都插上（当前只检测到 %zu 台）。\n", serials.size());
        return 1;
    }

    std::string Ls = (argc > 1) ? argv[1] : serials[0];
    std::string Rs = (argc > 2) ? argv[2] : serials[1];
    printf("左相机 = %s\n右相机 = %s\n", Ls.c_str(), Rs.c_str());

    HikCamera left, right;
    if (!left.open(Ls, 10000.0, 15.0, 60, false))  { std::fprintf(stderr, "左相机打开失败\n"); return 1; }
    if (!right.open(Rs, 10000.0, 15.0, 60, false)) { std::fprintf(stderr, "右相机打开失败\n"); return 1; }

    int start_idx = next_index();
    int idx = start_idx;
    int saved = 0;
    bool auto_mode = true;
    bool have_last = false;
    cv::Point2f last_center;
    float last_roll = 0.0f;

    char cwd[4096];
    if (getcwd(cwd, sizeof(cwd)))
        printf("图片将保存到: %s/left 和 %s/right\n", cwd, cwd);
    printf("提示：把棋盘格拿近(0.5~2m)、整块板在两台相机里都看得到；\n"
           "      右上角显示 L=OK/R=OK 才会自动存；一直 NO 就换方向、调距离和光线。\n");

    const char * win = "stereo capture  [a]自动/手动 [s]存 [d]删 [q]退";
    cv::namedWindow(win, cv::WINDOW_NORMAL);

    cv::Mat L, R;
    int frame_counter = 0;
    std::vector<cv::Point2f> cL, cR;
    cv::Size szL, szR;
    bool okL = false, okR = false;
    cv::Point2f center(0, 0);
    float roll = 0.0f;

    while (true)
    {
        std::uint64_t tsL = 0, tsR = 0;
        std::uint32_t fnL = 0, fnR = 0;
        bool okLg = left.grab(L, tsL, fnL, 1000);
        bool okRg = right.grab(R, tsR, fnR, 1000);
        if (!okLg || !okRg)
        {
            std::fprintf(stderr, "抓帧失败 L=%d R=%d\n", (int)okLg, (int)okRg);
            continue;
        }

        // 0.5x 预览图
        cv::Mat sL, sR;
        cv::resize(L, sL, cv::Size(), 0.5, 0.5);
        cv::resize(R, sR, cv::Size(), 0.5, 0.5);

        // 每 4 帧检测一次（省 CPU，预览不卡）；其余帧复用上次检测结果
        bool fresh = false;
        if ((frame_counter++ % 4) == 0)
        {
            cv::Mat Lg, Rg;
            cv::cvtColor(sL, Lg, cv::COLOR_BGR2GRAY);
            cv::cvtColor(sR, Rg, cv::COLOR_BGR2GRAY);
            okL = detect_board(Lg, cL, szL);
            okR = detect_board(Rg, cR, szR);
            fresh = true;
        }
        bool ok = okL && okR;

        if (ok && fresh)   // 只在刚检测到的帧算 center/roll
        {
            center = cv::Point2f(0, 0);
            for (auto & p : cL) center += p;
            center /= static_cast<float>(cL.size());
            cv::Point2f p0 = cL[0], p1 = cL[szL.width - 1];
            roll = std::atan2(p1.y - p0.y, p1.x - p0.x);
        }

        if (okL) cv::drawChessboardCorners(sL, szL, cL, true);
        if (okR) cv::drawChessboardCorners(sR, szR, cR, true);
        cv::Mat show;
        cv::hconcat(sL, sR, show);

        char status[160];
        std::snprintf(status, sizeof(status), "%s 已存%d  L=%s R=%s",
                      auto_mode ? "[自动]" : "[手动]", saved,
                      okL ? "OK" : "NO", okR ? "OK" : "NO");
        cv::putText(show, status, {10, 28}, cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
        cv::imshow(win, show);

        bool save_now = false;
        if (fresh && ok && auto_mode)
        {
            if (!have_last)
            {
                save_now = true;
            }
            else
            {
                float d  = cv::norm(center - last_center);
                float da = std::fabs(angle_diff(roll, last_roll));
                if (d > 50.0f || da > 0.35f) save_now = true;   // 0.5x 坐标下 50px ≈ 原图 100px
            }
        }

        int key = cv::waitKey(1);
        if (key == 'q' || key == 27) break;
        if (key == 'a') { auto_mode = !auto_mode; printf("切到 %s 模式\n", auto_mode ? "自动" : "手动"); }
        if (key == 's' || key == ' ')
        {
            if (ok) save_now = true;
            else printf("左右未同时检测到棋盘格，不存（L=%s R=%s）\n", okL ? "OK" : "NO", okR ? "OK" : "NO");
        }
        if (key == 'd')
        {
            if (idx > start_idx)
            {
                --idx;
                char n[64];
                std::snprintf(n, sizeof(n), "left/%04d.png", idx);  std::remove(n);
                std::snprintf(n, sizeof(n), "right/%04d.png", idx); std::remove(n);
                --saved; have_last = false;
                printf("已删除最后一张，下一张存到 %04d\n", idx);
            }
        }

        if (save_now)
        {
            char n[64];
            std::snprintf(n, sizeof(n), "left/%04d.png", idx);  cv::imwrite(n, L);
            std::snprintf(n, sizeof(n), "right/%04d.png", idx); cv::imwrite(n, R);
            printf("已存 %04d 对（共 %d 对）\n", idx, saved + 1);
            last_center = center; last_roll = roll; have_last = true;
            ++idx; ++saved;
        }
    }

    left.close();
    right.close();
    printf("结束，共存 %d 对到 left/ right/\n", saved);
    return 0;
}
