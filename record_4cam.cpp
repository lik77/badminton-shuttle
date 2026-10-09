// record_4cam.cpp —— 四相机同时录视频（采集训练数据用）
// 用法: ./record_4cam [输出目录，默认当前目录]
// 输出: wideL.avi / wideR.avi / teleL.avi / teleR.avi（各相机一路，MJPG 编码）
// 按 q 或 Esc 停止录制。
#include <cstdio>
#include <ctime>
#include <string>
#include <sys/stat.h>
#include <opencv2/opencv.hpp>
#include "rigs.hpp"

using namespace auto_aim;

int main(int argc, char ** argv)
{
    // 每次运行加时间戳目录，避免覆盖之前录的
    char ts[64];
    std::time_t now = std::time(nullptr);
    std::strftime(ts, sizeof(ts), "%Y%m%d_%H%M%S", std::localtime(&now));
    const std::string base = (argc > 1) ? argv[1] : "record";
    const std::string outdir = base + "_" + ts;
    mkdir(outdir.c_str(), 0777);  // 自动建输出目录
    // 视频帧率：设成【实际录制帧率】才不加速。4 台相机轮流抓+写+预览实际约 15~25fps，
    // 这里用 20；设成相机上限 120 反而会播得更快。
    const double fps = 20.0;

    RigSystem rigs;
    if (!rigs.load("configs/rigs.yaml"))
    {
        std::fprintf(stderr, "[record] 加载 configs/rigs.yaml 失败\n");
        return 1;
    }

    RigFrames F;
    if (!rigs.grab(F))
    {
        std::fprintf(stderr, "[record] 抓不到帧，检查相机/触发/带宽\n");
        return 1;
    }
    const cv::Size sz = F.wide.left.size();

    // 用 MJPG + AVI（OpenCV 内置编码器，不依赖 GStreamer/FFmpeg，最稳）
    const int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
    cv::VideoWriter wWL(outdir + "/wideL.avi", fourcc, fps, sz);
    cv::VideoWriter wWR(outdir + "/wideR.avi", fourcc, fps, sz);
    cv::VideoWriter wTL(outdir + "/teleL.avi", fourcc, fps, sz);
    cv::VideoWriter wTR(outdir + "/teleR.avi", fourcc, fps, sz);
    if (!wWL.isOpened() || !wWR.isOpened() || !wTL.isOpened() || !wTR.isOpened())
    {
        std::fprintf(stderr, "[record] 创建视频文件失败（检查输出目录是否可写）\n");
        return 1;
    }

    std::printf("[record] 开始录制到 %s/ ，按 q 停止\n", outdir.c_str());
    long frames = 0;
    while (true)
    {
        if (!rigs.grab(F))
        {
            std::fprintf(stderr, "[record] 抓帧失败\n");
            continue;
        }

        wWL.write(F.wide.left);
        wWR.write(F.wide.right);
        wTL.write(F.tele.left);
        wTR.write(F.tele.right);
        ++frames;

        cv::Mat a, b, c, d, top, bot, grid;
        cv::resize(F.wide.left, a, {640, 480});
        cv::resize(F.wide.right, b, {640, 480});
        cv::resize(F.tele.left, c, {640, 480});
        cv::resize(F.tele.right, d, {640, 480});
        cv::hconcat(a, b, top);
        cv::hconcat(c, d, bot);
        cv::vconcat(top, bot, grid);
        cv::putText(grid, "REC  [q]=stop", {10, grid.rows - 12}, cv::FONT_HERSHEY_SIMPLEX, 0.8,
                    cv::Scalar(0, 0, 255), 2);
        cv::imshow("record 4cam", grid);
        int key = cv::waitKey(1);
        if (key == 'q' || key == 27) break;
    }

    wWL.release();
    wWR.release();
    wTL.release();
    wTR.release();
    std::printf("[record] 完成，共录 %ld 帧，存到 %s/\n", frames, outdir.c_str());
    return 0;
}
