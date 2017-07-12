// @file bboxnms_gpu.cu
// @brief Bounding Box non maximum supression, heavily based 
// on Shaoqing Ren's Faster R-CNN implementation which 
// can be found here: 
// https://github.com/ShaoqingRen/faster_rcnn/blob/master/functions/nms
// @author Samuel Albanie

/*
Copyright (C) 2017- Samuel Albanie.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "bboxnms.hpp"
#include "../data.hpp"
#include <assert.h>
#include <float.h>
#include <cstdio>
#include <algorithm>
#include <math.h>
#include <string.h>
#include <map>
#include <vector>

/* ------------------------------------------------------------ */
/*                                                      kernels */
/* ------------------------------------------------------------ */

enum {
  XMIN = 0,
  YMIN,
  XMAX,
  YMAX,
} ;



namespace vl { namespace impl {

    template<typename T>
    struct bboxnms<vl::VLDT_GPU,T>
    {

    static vl::ErrorCode
    forward(Context& context,
            std::vector<int> &output,
            T const* boxes,
            float overlap, 
            size_t num_boxes) 
{
    printf("gpu version running\n") ;
    printf("vl_bboxnms: overlap: %d\n", overlap) ;
    printf("vl_bboxnms: num_boxes: %d\n", num_boxes) ;
    return VLE_Success ;
   }
 } ;
} } // namespace vl::impl

template struct vl::impl::bboxnms<vl::VLDT_GPU, float> ;

#ifdef ENABLE_DOUBLE
template struct vl::impl::bboxnms<vl::VLDT_GPU, double> ;
#endif
