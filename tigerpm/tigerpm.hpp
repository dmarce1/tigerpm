/*
 * tigerfmm.hpp
 *
 *  Created on: Jun 23, 2021
 *      Author: dmarce1
 */

#ifndef TIGERFMM_HPP_
#define TIGERFMM_HPP_


#ifndef __CUDACC__
#ifdef USE_HPX
#include <hpx/hpx.hpp>
#else
#include <hpx_lite/hpx/hpx_lite.hpp>
#endif
#endif

#include <stdio.h>

//#define CHECK_BOUNDS
#define FORCE_TEST
//#define SORT_TEST

#define TREEPM_BLOCK_SIZE 64
#define TREEPM_OVERSUBSCRIPTION 4
#define SINK_BUCKET_SIZE 64
#define SOURCE_BUCKET_SIZE 32
#define MAX_RUNG 32
#define NINTERP 6
#define NCELLS 27


#define NCHILD 2
#define PHI_BW 3
#define NDIM 3

#define XDIM 0
#define YDIM 1
#define ZDIM 2

#define CHAIN_BW 1


#define MAX_PARTS_PER_MSG (4*1024*1024)

#ifndef __CUDACC__
using spinlock_type = hpx::lcos::local::spinlock;
using mutex_type = hpx::lcos::local::mutex;
using shared_mutex_type = hpx::lcos::local::shared_mutex;
#endif


#define PRINT(...) print(__VA_ARGS__)

template<class ...Args>
#ifdef __CUDA_ARCH__
__device__
#endif
inline void print(const char* fmt, Args ...args) {
	printf(fmt, args...);
#ifndef __CUDA_ARCH__
	fflush(stdout);
#endif
}

#include <tigerpm/containers.hpp>
#endif /* TIGERFMM_HPP_ */
