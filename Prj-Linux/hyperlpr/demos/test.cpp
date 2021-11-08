#include <iostream>
#include <vector>
#include <filesystem>
#include <chrono>
#include "opencv2/core.hpp"
#include "spdlog/spdlog.h"
#include "Pipeline.h"
#include "PlateDetection.h"

using namespace std;
using namespace cv;
using namespace spdlog;
namespace fs = std::filesystem;
using namespace std::chrono;
using std::chrono::high_resolution_clock;

const size_t run_times = 100;

void test_pipeline(const char *img_path)
{
    pr::PipelinePR prc("model/cascade.xml",
                       "model/HorizonalFinemapping.prototxt", "model/HorizonalFinemapping.caffemodel",
                       "model/Segmentation.prototxt", "model/Segmentation.caffemodel",
                       "model/CharacterRecognization.prototxt", "model/CharacterRecognization.caffemodel",
                       "model/SegmenationFree-Inception.prototxt", "model/SegmenationFree-Inception.caffemodel");

    Mat img = imread(img_path);
    if (img.empty())
    {
        info("load image fail {}", img_path);
        return;
    }

    pr::PlateDetection plateDetection("model/cascade.xml");
    std::vector<pr::PlateInfo> plates;
    auto begin = high_resolution_clock::now();

    for (size_t i = 0; i < run_times; i++)
    {
        plateDetection.plateDetectionRough(img, plates);
        // prc.RunPiplineAsImage(img, pr::SEGMENTATION_FREE_METHOD);
    }

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - begin);
    cout << "run " << run_times << " times, cost time(ms): " << duration.count() << " per frame time(ms): " << duration.count() / run_times << endl;

    std::vector<pr::PlateInfo> res = prc.RunPiplineAsImage(img, pr::SEGMENTATION_FREE_METHOD);

    for (auto &st : res)
    {
        if (st.confidence > 0.25)
        {
            info("[{},{}]", st.getPlateName(), st.confidence);
            cv::Rect region = st.getPlateRect();
            cv::rectangle(img, cv::Point(region.x, region.y), cv::Point(region.x + region.width, region.y + region.height), cv::Scalar(255, 255, 0), 2);
        }
    }

    fs::path path(img_path);
    path.replace_filename("_" + path.filename().string());
    path.replace_extension("jpg");
    imwrite(path.c_str(), img);
}

int main(int argc, char *argv[])
{
    spdlog::info("test run ...");

    if (argc < 2)
    {
        spdlog::info("no image path");
        return 1;
    }

    const char *img_path = argv[1];

    test_pipeline(img_path);

    return 0;
}