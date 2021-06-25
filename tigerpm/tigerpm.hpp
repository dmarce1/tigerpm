/*
 * tigerfmm.hpp
 *
 *  Created on: Jun 23, 2021
 *      Author: dmarce1
 */

#ifndef TIGERFMM_HPP_
#define TIGERFMM_HPP_


#include <hpx/hpx.hpp>
#include <stdio.h>


#define NDIM 3

using spinlock_type = hpx::lcos::local::spinlock;
using mutex_type = hpx::lcos::local::mutex;
using shared_mutex_type = hpx::lcos::local::shared_mutex;


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
