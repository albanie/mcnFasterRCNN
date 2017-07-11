// vl_nnbboxnms.cu
// brief GPU nms block MEX wrapper
// author Samuel Albanie 

/*
Copyright (C) 2017 Samuel Albanie
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/


#include <bits/mexutils.h>
#include <bits/datamex.hpp>
#include "bits/nnbboxnms.hpp"

#if ENABLE_GPU
#include <bits/datacu.hpp>
#endif

#include <assert.h>

/* option codes */
enum {
  opt_verbose = 0,
} ;

/* options */
VLMXOption  options [] = {
  {"Verbose",         0,   opt_verbose          },
  {0,                 0,   0                    }
} ;

/* ---------------------------------------------------------------- */
/*                                                          Context */
/* ---------------------------------------------------------------- */

vl::MexContext context ;

/*
 Resetting the context here resolves a crash when MATLAB quits and
 the ~Context function is implicitly called on unloading the MEX file.
 */
void atExit()
{
  context.clear() ;
}

/* ---------------------------------------------------------------- */
/*                                                       MEX driver */
/* ---------------------------------------------------------------- */

enum {
  IN_BOXES = 0, IN_OVERLAP, IN_END
} ;

enum {
  OUT_RESULT = 0, OUT_END
} ;

void mexFunction(int nout, mxArray *out[],
                 int nin, mxArray const *in[])
{
  int verbosity = 0 ;
  int opt ;
  int next = IN_END ;
  mxArray const *optarg ;

  /* -------------------------------------------------------------- */
  /*                                            Check the arguments */
  /* -------------------------------------------------------------- */

  if (nin < 2) {
    mexErrMsgTxt("There are less than two arguments.") ;
  }

  // backwards mode is not yet supported for nms
  next = 2 ;

  while ((opt = vlmxNextOption (in, nin, options, &next, &optarg)) >= 0) {
    switch (opt) {
      case opt_verbose :
        ++ verbosity ;
        break ;

      default: 
        break ;
    }
  }

  vl::MexTensor boxes(context) ;

  boxes.init(in[IN_BOXES]) ;
  boxes.reshape(2) ;

  float overlap = (float)mxGetScalar(in[IN_OVERLAP]) ;

  /* check for appropriate input box shape */
  int box_dims = boxes.getWidth() ;
  if (box_dims != 5) {
    vlmxError(VLMXE_IllegalArgument, "BOXES should have shape N x 5.") ;
  }

  /* Create output buffers */
  vl::MexTensor output(context) ;
  vl::DataType dataType = boxes.getDataType() ;
  vl::TensorShape outputShape = vl::TensorShape(boxes.getHeight(), 1, 1, 1) ;
  output.initWithZeros(vl::VLDT_CPU, dataType, outputShape) ;

  if (verbosity > 0) {
    mexPrintf("vl_nnbboxnms: mode %s; %s\n",  
            (boxes.getDeviceType()==vl::VLDT_GPU)?"gpu":"cpu", "forward") ;
        vl::print("vl_nnbboxnms: boxes: ", boxes) ;
       // mexPrintf("vl_nnbboxnms: overlap: %d\n", overlap) ;
        vl::print("vl_nnbboxnms: output: ", output) ;
      }
      /* -------------------------------------------------------------- */
      /*                                                    Do the work */
      /* -------------------------------------------------------------- */

      vl::ErrorCode error ;
      error = vl::nnbboxnms_forward(context, 
                                    output, 
                                    boxes, 
                                    overlap) ;

  /* -------------------------------------------------------------- */
  /*                                                         Finish */
  /* -------------------------------------------------------------- */

  if (error != vl::VLE_Success) {
    mexErrMsgTxt(context.getLastErrorMessage().c_str()) ;
  }
  out[OUT_RESULT] = output.relinquish() ;
}
