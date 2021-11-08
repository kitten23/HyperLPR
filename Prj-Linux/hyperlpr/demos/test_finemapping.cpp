#include <iostream>
#include "FineMapping.h"

using namespace std;

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        cout << "no image path" << endl;
        return 1;
    }

    cv::Mat image = cv::imread(argv[1]);
    cv::Mat image_finemapping = pr::FineMapping::FineMappingVertical(image);
    pr::FineMapping finemapper = pr::FineMapping("model/HorizonalFinemapping.prototxt", "model/HorizonalFinemapping.caffemodel");
    image_finemapping = finemapper.FineMappingHorizon(image_finemapping, 0, -3);
    cv::imwrite("res/finemapping_result.jpg", image_finemapping);
    cv::imshow("image", image_finemapping);
    cv::waitKey(0);

    return 0;
}
