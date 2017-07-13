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
// #include <bits/mexutils.h>

#include <map>
#include <vector>

// division + round-up
#define DIVROUNDUP(x,y) ((x)/(y)+((x)%(y)>0))

/* ------------------------------------------------------------ */
/*                                                      kernels */
/* ------------------------------------------------------------ */

enum {
  XMIN = 0, YMIN, XMAX, YMAX,
} ;


// set number of threads per block (should be at least 64)
int const BLOCKSIZE = (sizeof(unsigned long long) * 8) ;

// compute intersection over union on the gpu
template <typename T>
__device__ inline float jaccard(T const * const a, T const * const b)
{
    T left = max(a[XMIN], b[XMIN]) ; 
    T right = min(a[XMAX], b[XMAX]) ;
    T top = max(a[YMIN], b[YMIN]) ;
    T bottom = min(a[YMAX], b[YMAX]) ;
    T width = max(right - left + 1, 0.f) ; 
    T height = max(bottom - top + 1, 0.f) ;
    T intersection = width * height ;
    T aArea = (a[XMAX] - a[XMIN] + 1) * (a[YMAX] - a[YMIN] + 1) ;
    T bArea = (b[XMAX] - b[XMIN] + 1) * (b[YMAX] - b[YMIN] + 1) ;
    return intersection / (aArea + bArea - intersection) ;
}

template <typename T>
__global__ void nmsKernel(const int numBoxes, 
                           const float overlapThresh, 
                           const T *boxes, 
                           unsigned long long *mask, 
                           const int colBlocks)
   {
    const int rowIdx = blockIdx.y ; 
    const int colIdx = blockIdx.x ;
    const int numRows = min(numBoxes - rowIdx * BLOCKSIZE, BLOCKSIZE) ; 
    const int numCols = min(numBoxes - colIdx * BLOCKSIZE, BLOCKSIZE) ;

    // all blocks in the same column of the block-grid will process the same
    // set of boxes
    int offset = BLOCKSIZE * colIdx ;

    // define shared memory for all the boxes processed by the current block
    __shared__ float blockBoxes[BLOCKSIZE * 5] ;

    // load bounding boxes and scores for current block into shared memory
    if (threadIdx.x < numCols)
    {
        blockBoxes[threadIdx.x*5 + 0] = boxes[(offset + threadIdx.x)*5 + 0] ;
        blockBoxes[threadIdx.x*5 + 1] = boxes[(offset + threadIdx.x)*5 + 1] ;
        blockBoxes[threadIdx.x*5 + 2] = boxes[(offset + threadIdx.x)*5 + 2] ;
        blockBoxes[threadIdx.x*5 + 3] = boxes[(offset + threadIdx.x)*5 + 3] ;
        blockBoxes[threadIdx.x*5 + 4] = boxes[(offset + threadIdx.x)*5 + 4] ;
    }

    // ensure that all threads in the block will have access to all boxes 
    // assigned to that block   
    __syncthreads() ;

    // process
    if (threadIdx.x < numRows)
    {
        const int boxIdx = BLOCKSIZE * rowIdx + threadIdx.x ;
        const T *currBox = boxes + boxIdx * 5 ;

        // use a bit mask to store box overlaps above the threshold
        unsigned long long tt = 0 ;

        // if current block lies on the diagonal of the grid, apply offset
        // (thi is to prevent a box from later being removed for having 
        // overlap with itself)
        int start = 0 ;
        if (rowIdx == colIdx) start = threadIdx.x + 1 ;

        // compare the current box against every other box in its block and
        // track its index if its overlap exceeds the threshold
        for (int ii = start; ii < numCols; ii++)
        {
            if (jaccard(currBox, blockBoxes + ii*5) > overlapThresh)
            {
                printf("overlap: %g for box %d vs offset %d\n", 
                        jaccard(currBox, blockBoxes + ii*5), boxIdx, ii)  ;
                printf("using threshold %g\n", overlapThresh) ;
                tt |= 1ULL << ii ;
            }
        }
        mask[boxIdx * colBlocks + colIdx] = tt ;
    }
}



namespace vl { namespace impl {

    template<typename T>
    struct bboxnms<vl::VLDT_GPU,T>
    {

    static vl::ErrorCode
    forward(Context& context,
            std::vector<int> &output,
            T const* boxes,
            float overlapThresh, 
            size_t numBoxes,
            int &numKept) 
    {
    const int colBlocks = DIVROUNDUP(numBoxes, BLOCKSIZE);

    // Allocate memory to hold the nms results mask 
    unsigned long long *mask = NULL;
    int MASK_ARRAY_BYTES = sizeof(unsigned long long) * numBoxes * colBlocks ;
    cudaMalloc(&mask, MASK_ARRAY_BYTES);
    
    // we will only use the x-dim on each thread block
    dim3 threads(BLOCKSIZE) ; 

    // the thread blocks are arranged as an square grid
    dim3 blocks(DIVROUNDUP(numBoxes, BLOCKSIZE), DIVROUNDUP(numBoxes, BLOCKSIZE)) ;

    nmsKernel<T><<<blocks,threads>>>(numBoxes, overlapThresh, boxes, mask, colBlocks) ;

    // use mask_h to hold results and copy back from device
    std::vector<unsigned long long> mask_h(numBoxes * colBlocks);
    cudaMemcpy(&mask_h[0], mask, MASK_ARRAY_BYTES, cudaMemcpyDeviceToHost);

    // use `remv` to keep track of which blocks have been processed 
    std::vector<unsigned long long> remv(colBlocks);
    memset(&remv[0], 0, sizeof(unsigned long long) * colBlocks);

    for (int ii = 0; ii < numBoxes; ii++)
    {
        int blockNum = ii / BLOCKSIZE ;
        int inblock = ii % BLOCKSIZE ;

        // check that the current box has not yet been "removed"
        if (!(remv[blockNum] & (1ULL << inblock)))
        {
            output[numKept] = ii ;  // store box index
            numKept += 1 ;
            unsigned long long *p = &mask_h[0] + ii * colBlocks ;

            // remove boxes with high overlap following the current one
            for (int jj = blockNum; jj < colBlocks; jj++)
            {
                remv[jj] |= p[jj] ;
            }
        }
    }
    cudaFree(mask);  

    return VLE_Success ;
   }
 } ;
} } // namespace vl::impl

template struct vl::impl::bboxnms<vl::VLDT_GPU, float> ;

#ifdef ENABLE_DOUBLE
template struct vl::impl::bboxnms<vl::VLDT_GPU, double> ;
#endif
