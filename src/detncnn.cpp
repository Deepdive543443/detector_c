#include <stdlib.h>
#include "net.h"
#include "detector/nanodetplus.hpp"
#include "detector/fastestdet.hpp"
#include "detector/rtmdet.hpp"
#include "detncnn.h"

#ifdef __cplusplus
extern "C" {
#endif

void *det_init(DET_PARAM_T *opt)
{
    Detector *net;
    switch (opt->model_type) {
        case DET_NANODETPLUS:
            net = (Detector *)new NanoDetPlus;
            break;

        case DET_FASTESTDET:
            net = (Detector *)new FastestDet;
            break;

        case DET_RTMDET:
            net = (Detector *)new RTMDet;
            break;

        default:
            return NULL;
    }
    net->load(opt);
    return (void *)net;
}

int det_exit(void *ctx)
{
    if (ctx) {
        delete (Detector *)ctx;
        return 1;
    }
    return 0;
}

int det_detect(void *ctx, unsigned char *rgb, int h, int w, DET_OBJ_T *output, int *out_len)
{
    std::vector<DET_OBJ_T> objects;
    Detector *det = (Detector *)ctx;

    if (det->detect(rgb, w, h, objects)) {
        *out_len = objects.size() <= DET_OBJ_BUFSIZE ? objects.size() : DET_OBJ_BUFSIZE;
        memcpy(output, &objects[0], *out_len * sizeof(DET_OBJ_T));
        return 1;
    }
    return 0;
}

int det_detect_nv12(void *ctx, unsigned char *nv12, int h, int w, DET_OBJ_T *output, int *out_len)
{
    unsigned char *rgb = (unsigned char *)malloc(h * w * 3);
    ncnn::yuv420sp2rgb_nv12(nv12, w, h, rgb);
    int ret = det_detect(ctx, rgb, h, w, output, out_len);
    free(rgb);
    return ret;
}

int det_draw_boxxes(unsigned char *rgb, int h, int w, DET_OBJ_T *output, int *out_len)
{
    std::vector<DET_OBJ_T> objs;
    for (int i = 0; i < *out_len; i++) objs.push_back(output[i]);
    return detncnn::draw_boxxes(rgb, w, h, objs);
}

#ifdef __cplusplus
}  // extern "C"
#endif