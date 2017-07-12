// @file bboxnms_cpu.cu
// @brief Bounding Box non maximum supression, based // on Shaoqing Ren's Faster R-CNN implementation which 
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

namespace vl { namespace impl {

  template<typename T>
  struct bboxnms<vl::VLDT_CPU,T>
  {

    static vl::ErrorCode
    forward(Context& context,
            T* output,
            T const* boxes,
            float overlap, 
            size_t num_boxes)
    {
      printf("cpu version running\n") ;
      printf("vl_bboxnms: overlap: %d\n", overlap) ;
      printf("vl_bboxnms: num_boxes: %d\n", num_boxes) ;
      return VLE_Success ;
   }
 } ;
} } // namespace vl::impl

template struct vl::impl::bboxnms<vl::VLDT_CPU, float> ;

#ifdef ENABLE_DOUBLE
template struct vl::impl::bboxnms<vl::VLDT_CPU, double> ;
#endif
