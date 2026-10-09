// 单相机冒烟测试（独立文件，不修改 shuttle/ 现有代码）
// 复用 src/hik_camera.cpp 的 HikCamera 封装。
#include <cstdio>
#include <chrono>
#include <opencv2/imgcodecs.hpp>
#include "hik_camera.hpp"

using namespace auto_aim;

int main()
{
    HikCamera cam;
    // 空序列号 = 打开枚举到的第一台；曝光 2000us、增益 16.0、自由运行 90fps
    if (!cam.open("0", 4000.0, 13.0, 90, false))
    {
        std::fprintf(stderr, "[test] 打开相机失败（检查 USB3 线 / 是否被 MVS 客户端占用 / 权限）\n");
        return 1;
    }

    cv::Mat bgr;
    std::uint64_t ts = 0;
    std::uint32_t fn = 0;

    const int N = 100;
    int ok = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < N; ++i)
    {
        if (cam.grab(bgr, ts, fn, 2000))
        {
            ++ok;
            if (i == 0)
                std::printf("首帧: %dx%d  dev_ts=%llu ns  frame_num=%u\n",
                            bgr.cols, bgr.rows, (unsigned long long)ts, (unsigned)fn);
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();

    std::printf("抓取 %d/%d 成功, 耗时 %.2fs, 平均 %.1f fps\n", ok, N, sec, ok / sec);
    if (ok > 0)
        std::printf("末帧: %dx%d  dev_ts=%llu ns  frame_num=%u\n",
                    bgr.cols, bgr.rows, (unsigned long long)ts, (unsigned)fn);

    if (!bgr.empty())
    {
        cv::imwrite("test_camera.png", bgr);
        std::printf("已保存 test_camera.png（可打开看是否成像）\n");
    }

    cam.close();
    std::printf("[test] 完成\n");
    return ok > 0 ? 0 : 1;
}
