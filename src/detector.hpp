#ifndef DETECTOR_H
#define DETECTOR_H

#include "net.h"
#include "detncnn.h"

namespace detncnn {
int draw_boxxes(unsigned char *rgb, int width, int height, std::vector<DET_OBJ_T> &objects);
}

class Detector {
   public:
    Detector();
    virtual ~Detector();

    void qsort_descent_inplace(std::vector<DET_OBJ_T> &objects, int left, int right);
    void qsort_descent_inplace(std::vector<DET_OBJ_T> &objects);
    void nms_sorted_bboxes(const std::vector<DET_OBJ_T> &objects, std::vector<int> &picked, float nms_threshold);

    virtual int load(DET_PARAM_T *opt);
    virtual int detect(unsigned char *rgb, int width, int height, std::vector<DET_OBJ_T> &objects);

    ncnn::Net                   net;
    int                         target_size;
    int                         stream_w;
    int                         stream_h;
    float                       mean_vals[3];
    float                       norm_vals[3];
    float                       prob_threshold;
    float                       nms_threshold;
    ncnn::UnlockedPoolAllocator blob_pool_allocator;
    ncnn::PoolAllocator         workspace_pool_allocator;
};

#endif  // DETECTOR_H