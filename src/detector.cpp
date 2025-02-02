#include "detector.hpp"

static const char *s_class_names[] = {
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

static uint32_t s_color_list[] = {
    0x1852d8ff, 0x1fb0ecff, 0x8d2e7dff, 0x2fab76ff, 0xedbd4cff, 0x2e13eeff, 0x4c4c4cff, 0x999999ff, 0x0000ffff,
    0x007fffff, 0x00bebeff, 0x00ff00ff, 0xff0000ff, 0xff00aaff, 0x005454ff, 0x00aa54ff, 0x00ff54ff, 0x0054aaff,
    0x00aaaaff, 0x00ffaaff, 0x0054ffff, 0x00aaffff, 0x00ffffff, 0x7f5400ff, 0x7faa00ff, 0x7fff00ff, 0x7f0054ff,
    0x7f5454ff, 0x7faa54ff, 0x7fff54ff, 0x7f00aaff, 0x7f54aaff, 0x7faaaaff, 0x7fffaaff, 0x7f00ffff, 0x7f54ffff,
    0x7faaffff, 0x7fffffff, 0xff5400ff, 0xffaa00ff, 0xffff00ff, 0xff0054ff, 0xff5454ff, 0xffaa54ff, 0xffff54ff,
    0xff00aaff, 0xff54aaff, 0xffaaaaff, 0xffffaaff, 0xff00ffff, 0xff54ffff, 0xffaaffff, 0x00002aff, 0x000054ff,
    0x00007fff, 0x0000aaff, 0x0000d4ff, 0x0000ffff, 0x002a00ff, 0x005400ff, 0x007f00ff, 0x00aa00ff, 0x00d400ff,
    0x00ff00ff, 0x2a0000ff, 0x540000ff, 0x7f0000ff, 0xaa0000ff, 0xd40000ff, 0xff0000ff, 0x000000ff, 0x242424ff,
    0x484848ff, 0x6d6d6dff, 0x919191ff, 0xb6b6b6ff, 0xdadadaff, 0xbc7100ff, 0xbcb650ff, 0x007f7fff};

#define DRAW_TEXT_SIZE   8
#define DRAW_TEXT_OFFSET 4
#define DRAW_FLAG_H      22
#define DRAW_FLAG_OFFSET 2

#define DRAW_FLAG_W   DRAW_TEXT_SIZE * 1.1
#define DRAW_FLAG_POS DRAW_FLAG_H + DRAW_FLAG_OFFSET
#define DRAW_TEXT_POS DRAW_FLAG_H + DRAW_TEXT_OFFSET
int detncnn::draw_boxxes(unsigned char *rgb, int width, int height, std::vector<DET_OBJ_T> &objects)
{
    for (size_t i = 0; i < objects.size(); i++) {
        char text[32];
        snprintf(text, 32, "%s %.1f%%", s_class_names[objects[i].label], objects[i].prob * 100);

        int flag_size = strlen(text) * DRAW_FLAG_W;

        ncnn::draw_rectangle_c3(rgb, width, height, objects[i].x, objects[i].y, objects[i].w, objects[i].h,
                                s_color_list[i], 3);
        ncnn::draw_rectangle_c3(rgb, width, height, objects[i].x - 1, objects[i].y - DRAW_FLAG_POS, flag_size,
                                DRAW_FLAG_H, s_color_list[i], -1);
        ncnn::draw_text_c3(rgb, width, height, text, (int)objects[i].x + 1, (int)objects[i].y - DRAW_TEXT_POS,
                           DRAW_TEXT_SIZE, 0);
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
