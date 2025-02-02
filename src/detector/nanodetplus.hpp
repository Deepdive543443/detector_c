// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#ifndef NANODETPLUS_H
#define NANODETPLUS_H

#include "detector.hpp"

class NanoDetPlus : private Detector {
   public:
    virtual int load(DET_PARAM_T *opt);
    virtual int detect(unsigned char *rgb, int width, int height, std::vector<DET_OBJ_T> &objects);
};

#endif  // NANODETPLUS_H
