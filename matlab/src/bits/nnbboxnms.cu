// file nnbboxnms.cu
// brief nms block
// author Samuel Albanie

/*
Copyright (C) 2017- Samuel Albanie
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "nnbboxnms.hpp"
#include "impl/bboxnms.hpp"

#if ENABLE_GPU
#include "datacu.hpp"
#endif

#include <cstdio>
#include <assert.h>

using namespace vl ;

/* ---------------------------------------------------------------- */
/*                                         bboxnms_forward */
/* ---------------------------------------------------------------- */

#define DISPATCH(deviceType,T) \
error = vl::impl::bboxnms<deviceType,T>::forward (context, \
(T*) output.getMemory(), \
(T const*) boxes.getMemory(), \
(float) overlap, \
(size_t) boxes.getWidth()) ;

#define DISPATCH2(deviceType) \
switch (dataType) { \
case VLDT_Float : DISPATCH(deviceType, float) ; \
break ; \
IF_DOUBLE(case VLDT_Double : DISPATCH(deviceType, double) ; \
break ;) \
default: assert(false) ; \
return VLE_Unknown ; \
}

vl::ErrorCode
vl::nnbboxnms_forward(vl::Context& context,
                      vl::Tensor output,
                      vl::Tensor boxes,
                      float overlap)
{
  vl::ErrorCode error = VLE_Success ;
  vl::DataType dataType = boxes.getDataType() ;
  
  switch (boxes.getDeviceType())
  {
    case vl::VLDT_CPU:
      printf("cpu version running\n") ;
      DISPATCH2(vl::VLDT_CPU) ;
      break ;

#if ENABLE_GPU
    case vl::VLDT_GPU:
      DISPATCH2(vl::VLDT_GPU) ;
    if (error == VLE_Cuda) {
      context.setError(context.getCudaHelper().catchCudaError("GPU")) ;
    }
    break;
#endif

    default:
      assert(false);
      error = vl::VLE_Unknown ;
      break ;
  }
  return context.passError(error, __func__);
}
