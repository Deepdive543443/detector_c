#include "detector.hpp"

static const char *g_class_names[] = {
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

static uint8_t g_color_list[80][3] = {
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
    {80, 182, 188},  {127, 127, 0},
};

#define DRAW_TEXT_SIZE  7
#define DRAW_FLAG_SIZE 18
int detncnn::draw_boxxes(unsigned char *rgb, int width, int height, std::vector<DET_OBJ_T> &objects)
{
    for (size_t i = 0; i < objects.size(); i++) {
        int rgba = (g_color_list[i][2] << 24) | (g_color_list[i][1] << 16) | (g_color_list[i][0] << 8) | 255;

        char text[32];
        snprintf(text, 32, "%s %.1f%%", g_class_names[objects[i].label], objects[i].prob * 100);

        ncnn::draw_rectangle_c3(rgb, width, height, objects[i].x, objects[i].y, objects[i].w, objects[i].h, rgba, 3);
        ncnn::draw_rectangle_c3(rgb, width, height, objects[i].x, objects[i].y, objects[i].w, DRAW_FLAG_SIZE, rgba, -1);
        ncnn::draw_text_c3(rgb, width, height, text, (int)objects[i].x + 1, (int)objects[i].y + 1, DRAW_TEXT_SIZE, 0);
    }
    return 1;
}

Detector::Detector()
{
    blob_pool_allocator.set_size_compare_ratio(0.f);
    workspace_pool_allocator.set_size_compare_ratio(0.f);
}

Detector::~Detector() { net.clear(); }

int Detector::load(DET_PARAM_T *opt) { return 0; }
int Detector::detect(unsigned char *rgb, int width, int height, std::vector<DET_OBJ_T> &objects) { return 0; }

static inline float intersection_area(const DET_OBJ_T &a, const DET_OBJ_T &b)
{
    float xA  = std::max(a.x, b.x);
    float yA  = std::max(a.y, a.y);
    float xB  = std::min(a.x + a.w, b.x + b.w);
    float yB  = std::min(a.y + a.h, b.y + b.h);
    return std::max(0.f, xB - xA) * std::max(0.f, yB - yA);
}

void Detector::qsort_descent_inplace(std::vector<DET_OBJ_T> &objects, int left, int right)
{
    int   i = left;
    int   j = right;
    float p = objects[(left + right) / 2].prob;

    while (i <= j) {
        while (objects[i].prob > p) i++;

        while (objects[j].prob < p) j--;

        if (i <= j) {
            // swap
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

void Detector::qsort_descent_inplace(std::vector<DET_OBJ_T> &objects)
{
    if (objects.empty()) return;
    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

void Detector::nms_sorted_bboxes(const std::vector<DET_OBJ_T> &objects, std::vector<int> &picked, float nms_threshold)
{
    picked.clear();

    const int n = objects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++) {
        areas[i] = objects[i].w * objects[i].h;
    }

    for (int i = 0; i < n; i++) {
        const DET_OBJ_T &a = objects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++) {
            const DET_OBJ_T &b = objects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold) keep = 0;
        }

        if (keep) picked.push_back(i);
    }
}
