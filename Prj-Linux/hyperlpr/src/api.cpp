
#include <stdint.h>
#include <memory>
#include "opencv2/core.hpp"

#include "Pipeline.h"

#define Api

using namespace std;
using namespace cv;
unique_ptr<pr::PipelinePR> pipeline;

#if defined(__cplusplus)
extern "C"
{
#endif // __cplusplus
    int const plate_max_length = 20;

    struct LprInfo
    {
        char plate[plate_max_length];
        // int plate_len;
        float conf;          // confidence - probability that the object was found correctly
        int x, y, w, h; // (x,y) - top-left corner, (w, h) - width & height of bounded box
    };

    Api int Init()
    {
        try
        {
            if (nullptr != pipeline)
            {
                return 0;
            }

            cout << "api Init" << endl;
            pipeline = make_unique<pr::PipelinePR>("model/cascade.xml",
                                                   "model/HorizonalFinemapping.prototxt", "model/HorizonalFinemapping.caffemodel",
                                                   "model/Segmentation.prototxt", "model/Segmentation.caffemodel",
                                                   "model/CharacterRecognization.prototxt", "model/CharacterRecognization.caffemodel",
                                                   "model/SegmenationFree-Inception.prototxt", "model/SegmenationFree-Inception.caffemodel");

            return 0;
        }
        catch (const std::exception &e)
        {
            std::cerr << e.what() << '\n';
            return -1;
        }
    }

    Api int Dispose()
    {
        cout << "api Dispose" << endl;

        pipeline = nullptr;
        return 0;
    }

    Api int Recognise(int width, int height, uint8_t *data, int step, LprInfo *buf, int buf_len)
    {
        try
        {
            cout << "api Recognise " << width << " " << height << " " << step << endl;

            if (nullptr == pipeline)
            {
                return -2;
            }

            if (nullptr == data)
            {
                return -3;
            }

            int res_count = 0;
            Mat img(height, width, CV_8UC3, data, step);
            imwrite("_img.jpg", img);
            std::vector<pr::PlateInfo> res = pipeline->RunPiplineAsImage(img, pr::SEGMENTATION_FREE_METHOD);
            cout << "RunPiplineAsImage: " << res.size() << endl;

            for (auto &plate : res)
            {
                if (res_count >= buf_len)
                {
                    break;
                }

                auto &info = buf[res_count];
                // cout << "input info conf:" << info.conf << " " << info.w << endl;

                // auto p = plate.getPlateName().c_str();
                // cout << "plate " << p << endl;
                // buf[res_count].plate = p;
                plate.getPlateName().copy(info.plate, plate_max_length);
                info.conf = plate.confidence;
                cv::Rect region = plate.getPlateRect();
                info.x = region.x;
                info.y = region.y;
                info.w = region.width;
                info.h = region.height;

                res_count++;
            }

            cout << "RunPiplineAsImage res_count: " << res_count << endl;

            return res_count;
        }
        catch (const std::exception &e)
        {
            std::cerr << e.what() << '\n';
            return -1;
        }
    }

#if defined(__cplusplus)
}
#endif // __cplusplus
