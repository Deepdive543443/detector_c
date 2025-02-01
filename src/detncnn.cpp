#include <stdlib.h>
#include "net.h"
#include "detector/nanodetplus.h"
#include "detncnn.h"

#ifdef __cplusplus
extern "C" {
#endif

void *nanodet_plus_init(DET_PARAM_T *opt)
{
    Detector *nanodet_plus = (Detector *)new NanoDetPlus;
    nanodet_plus->load(opt);
    return (void *)nanodet_plus;
}

int det_exit(void *ctx)
{
    if (ctx) {
        delete (Detector *)ctx;
        return 1;
    }
    return 0;
}

int det_detect(void *ctx, unsigned char *rgb, int h, int w, DET_OBJ_T *output)
{
    std::vector<DET_OBJ_T> objects;
    Detector *det = (Detector *)ctx;
    det->detect(rgb, w, h, objects);
    for (size_t i = 0; i < objects.size() && i < DET_OBJ_BUFSIZE; i++) output[i] = objects[i];
    return 1;
}

int det_detect_nv12(void *ctx, unsigned char *nv12, int h, int w, DET_OBJ_T *output)
{
    unsigned char *rgb = (unsigned char *)malloc(h * w * 3);
    ncnn::yuv420sp2rgb_nv12(nv12, w, h, rgb);
    det_detect(ctx, rgb, h, w, output);
    free(rgb);
    return 1;
}

#ifdef __cplusplus
}  // extern "C"
#endif