/*
 * tigerfmm.hpp
 *
 *  Created on: Jun 23, 2021
 *      Author: dmarce1
 */

#ifndef TIGERFMM_HPP_
#define TIGERFMM_HPP_


#ifndef __CUDACC__
#include <hpx/hpx.hpp>
#endif

#include <stdio.h>

//#define FORCE_TEST
//#define SORT_TEST

#define NDIM 3

#define XDIM 0
#define YDIM 1
#define ZDIM 2

#define CHAIN_RATIO 4
#define CHAIN_BW 1

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

#endif /* TIGERFMM_HPP_ */
