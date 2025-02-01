#include "detector.h"

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