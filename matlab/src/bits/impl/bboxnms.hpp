// @file bboxnms.hpp
// @brief Non maximum supression
// @author Samuel Albanie

/*
Copyright (C) 2017- Samuel Albanie.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef VL_BBOXNMS_H
#define VL_BBOXNMS_H

#include "../data.hpp"
#include <cstddef>
#include <vector>

// defines the dispatcher for CUDA kernels:
namespace vl { namespace impl {

  template<vl::DeviceType dev, typename T>
  struct bboxnms {

    static vl::ErrorCode
    forward(Context& context,
            std::vector<int> &output,
            T const* boxes,
            float overlap,
            size_t num_boxes) ;

  } ;
} 

}

#endif /* defined(VL_BBOXNMS_H) */
