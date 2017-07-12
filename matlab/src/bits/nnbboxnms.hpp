// file nnbboxnms.hpp
// brief NMS block
// author Samuel Albanie 
/*
Copyright (C) 2017 Samuel Albanie
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef __vl__nnbboxnms__
#define __vl__nnbboxnms__

#include "data.hpp"
#include <stdio.h>

namespace vl {

  vl::ErrorCode
  nnbboxnms_forward(vl::Context& context,
                    vl::Tensor output,
                    vl::Tensor boxes,
                    float overlap) ;
}

#endif /* defined(__vl__nnbboxnms__) */
