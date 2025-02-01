#ifndef DETNCNN_H
#define DETNCNN_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float x;
    float y;
    float w;
    float h;
    int   label;
    float prob;
} DET_OBJ_T;

typedef struct {
    const char *model_path;
    int         stream_w;
    int         stream_h;
    float       prob_threshold;
    float       nms_threshold;
    int         use_gpu;
} DET_PARAM_T;

#define DET_OBJ_BUFSIZE 80

void *nanodet_plus_init(DET_PARAM_T *opt);
int   det_exit(void *ctx);
int   det_detect(void *ctx, unsigned char *rgb, int h, int w, DET_OBJ_T *output);
int   det_detect_nv12(void *ctx, unsigned char *nv12, int h, int w, DET_OBJ_T *output);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // DETNCNN_H