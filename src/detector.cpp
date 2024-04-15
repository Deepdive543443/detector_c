#include <string>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"  // opencv
#include "cpu.h"                        // ncnn
#include "detector.hpp"

namespace det {

static const char *class_names[] = {
    "person",         "bicycle",    "car",           "motorcycle",    "airplane",     "bus",           "train",
    "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",    "parking meter", "bench",
    "bird",           "cat",        "dog",           "horse",         "sheep",        "cow",           "elephant",
    "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",     "handbag",       "tie",
    "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball",  "kite",          "baseball bat",
    "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",       "wine glass",    "cup",
    "fork",           "knife",      "spoon",         "bowl",          "banana",       "apple",         "sandwich",
    "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",        "donut",         "cake",
    "chair",          "couch",      "potted plant",  "bed",           "dining table", "toilet",        "tv",
    "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",   "microwave",     "oven",
    "toaster",        "sink",       "refrigerator",  "book",          "clock",        "vase",          "scissors",
    "teddy bear",     "hair drier", "toothbrush"};

static const uint8_t color_list[80][3] = {
    {216, 82, 24},   {236, 176, 31},  {125, 46, 141},  {118, 171, 47},  {76, 189, 237},  {238, 19, 46},
    {76, 76, 76},    {153, 153, 153}, {255, 0, 0},     {255, 127, 0},   {190, 190, 0},   {0, 255, 0},
    {0, 0, 255},     {170, 0, 255},   {84, 84, 0},     {84, 170, 0},    {84, 255, 0},    {170, 84, 0},
    {170, 170, 0},   {170, 255, 0},   {255, 84, 0},    {255, 170, 0},   {255, 255, 0},   {0, 84, 127},
    {0, 170, 127},   {0, 255, 127},   {84, 0, 127},    {84, 84, 127},   {84, 170, 127},  {84, 255, 127},
    {170, 0, 127},   {170, 84, 127},  {170, 170, 127}, {170, 255, 127}, {255, 0, 127},   {255, 84, 127},
    {255, 170, 127}, {255, 255, 127}, {0, 84, 255},    {0, 170, 255},   {0, 255, 255},   {84, 0, 255},
    {84, 84, 255},   {84, 170, 255},  {84, 255, 255},  {170, 0, 255},   {170, 84, 255},  {170, 170, 255},
    {170, 255, 255}, {255, 0, 255},   {255, 84, 255},  {255, 170, 255}, {42, 0, 0},      {84, 0, 0},
    {127, 0, 0},     {170, 0, 0},     {212, 0, 0},     {255, 0, 0},     {0, 42, 0},      {0, 84, 0},
    {0, 127, 0},     {0, 170, 0},     {0, 212, 0},     {0, 255, 0},     {0, 0, 42},      {0, 0, 84},
    {0, 0, 127},     {0, 0, 170},     {0, 0, 212},     {0, 0, 255},     {0, 0, 0},       {36, 36, 36},
    {72, 72, 72},    {109, 109, 109}, {145, 145, 145}, {182, 182, 182}, {218, 218, 218}, {0, 113, 188},
    {80, 182, 188},  {127, 127, 0}};

static inline float intersection_area(const Object &a, const Object &b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

Detector::Detector()
{
    blob_pool_allocator.set_size_compare_ratio(0.f);
    workspace_pool_allocator.set_size_compare_ratio(0.f);

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    network.opt = ncnn::Option();

    network.opt.num_threads              = ncnn::get_big_cpu_count();
    network.opt.blob_allocator           = &blob_pool_allocator;
    network.opt.workspace_allocator      = &workspace_pool_allocator;
    network.opt.use_winograd_convolution = true;
    network.opt.use_sgemm_convolution    = true;
}

Detector::~Detector() {}

void Detector::info() const
{
    ncnn::Mat input(input_size, input_size, 3);

    const std::vector<const char *> &input_names  = network.input_names();
    const std::vector<const char *> &output_names = network.output_names();

    ncnn::Extractor extractor = network.create_extractor();
    printf("=======================\n");
    for (auto name : input_names) {
        printf("in_name: %s\nShape: C: %d H: %d W: %d\n\n", name, input.c, input.h, input.w);
        extractor.input(name, input);
    }

    for (auto name : output_names) {
        ncnn::Mat out;
        extractor.extract(name, out);
        printf("out_name: %s\nShape: C: %d H: %d W: %d\n\n", name, out.c, out.h, out.w);
    }
    printf("=======================\n");
}

void Detector::qsort_descent_inplace(std::vector<Object> &objects, int left, int right) const
{
    int   i = left;
    int   j = right;
    float p = objects[(left + right) / 2].prob;

    while (i <= j) {
        while (objects[i].prob > p) i++;

        while (objects[j].prob < p) j--;

        if (i <= j) {
            std::swap(objects[i], objects[j]);

            i++;
            j--;
        }
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (left < j) qsort_descent_inplace(objects, left, j);
        }
        #pragma omp section
        {
            if (i < right) qsort_descent_inplace(objects, i, right);
        }
    }
}

void Detector::qsort_descent_inplace(std::vector<Object> &objects) const
{
    if (objects.empty()) return;
    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

void Detector::draw_boxxes(cv::Mat &rgb, std::vector<Object> &objects) const
{
    int color_index = 0;

    for (size_t i = 0; i < objects.size(); i++) {
        const Object        &obj   = objects[i];
        const unsigned char *color = color_list[color_index % 19];
        color_index++;

        cv::Scalar cc(color[0], color[1], color[2]);

        cv::rectangle(rgb, obj.rect, cc, 2);

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int      baseLine   = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0) y = 0;
        if (x + label_size.width > rgb.cols) x = rgb.cols - label_size.width;

        cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)), cc, -1);
        cv::Scalar textcc = (color[0] + color[1] + color[2] >= 381) ? cv::Scalar(0, 0, 0) : cv::Scalar(255, 255, 255);
        cv::putText(rgb, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.5, textcc, 1);
    }
}

void Detector::nms(std::vector<Object> &objects, std::vector<int> &picked_box_idx, float thresh) const
{
    picked_box_idx.clear();
    const int n = objects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++) {
        areas[i] = objects[i].rect.width * objects[i].rect.height;
    }

    for (int i = 0; i < n; i++) {
        const Object &a = objects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked_box_idx.size(); j++) {
            const Object &b = objects[picked_box_idx[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked_box_idx[j]] - inter_area;

            // float IoU = inter_area / union_area
            if (inter_area / union_area > thresh) keep = 0;
        }

        if (keep) picked_box_idx.push_back(i);
    }
}

void Detector::load(int size, const char *path_param, const char *path_bin, int use_gpu)
{
#if NCNN_VULKAN
    network.opt.use_vulkan_compute = use_gpu;
#endif
    std::cout << "Base load function" << std::endl;
    return;
};

void Detector::detect(cv::Mat &rbg, std::vector<Object> &objects) const
{
    std::cout << "Base detect function" << std::endl;
    return;
};

}  // namespace det