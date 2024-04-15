#ifndef DETECTOR_HPP
#define DETECTOR_HPP

#include <vector>
#include "opencv2/core/core.hpp"
#include "net.h"

namespace det {

enum {
    NANODET_PLUS,
    FASTESTDET
};

typedef struct Object {
    cv::Rect_<float> rect;
    int              label;
    float            prob;
} Object;

class Detector {
   protected:
    // General
    int   input_size;
    float mean_vals[3];
    float norm_vals[3];
    float prob_thresh;
    float nms_thresh;

    ncnn::UnlockedPoolAllocator blob_pool_allocator;
    ncnn::PoolAllocator         workspace_pool_allocator;
    ncnn::Net                   network;

    void qsort_descent_inplace(std::vector<Object> &objects, int left, int right) const;
    void qsort_descent_inplace(std::vector<Object> &objects) const;
    void nms(std::vector<Object> &objects, std::vector<int> &picked_box_idx, float thresh) const;

   public:
    Detector();
    virtual ~Detector();

    void info() const;
    void draw_boxxes(cv::Mat &rgb, std::vector<Object> &objects) const;

    virtual void load(int size, const char *path_param, const char *path_bin, int use_gpu);
    virtual void detect(cv::Mat &rbg, std::vector<Object> &objects) const;
};

Detector *create_nanodet_plus();
Detector *create_fastestdet();

}  // namespace det

#endif  // DETECTOR_HPP