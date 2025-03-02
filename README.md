## Usage

C
```C
#include "detncnn.h"

DET_PARAM_T opt = {.model_type = DET_RTMDET,
                   .model_path = "rtmdet_m_8xb32-300e_coco",
                   .stream_w = width,
                   .stream_h = height};

void *det = det_init(&opt);
DET_OBJ_T buf[80];
int       out_len;

det_detect(det, pixels, pixel_h, pixel_w, buf, &out_len);
det_draw_boxxes(det, pixels, pixel_h, pixel_h, buf, &out_len);
```
C++
```C++
#include "detector/nanodetplus.hpp"

DET_PARAM_T opt = {.model_type = DET_NANODETPLUS,
                   .model_path = "nanodet-plus-m_416_int8",
                   .stream_w = width,
                   .stream_h = height};

NanoDetPlus nanodetplus(&opt);
std::vector<DET_OBJ_T> objects;

nanodetplus.detect(pixels, pixel_h, pixel_w, objects);
detncnn::draw_boxxes(pixels, pixel_h, pixel_w, objs);
```
