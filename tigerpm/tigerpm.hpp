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

#define SELF_PHI float(-35.0/16.0)

//#define CHECK_BOUNDS
#define FORCE_TEST
//#define SORT_TEST

#define CLOUD_W 4
#define USE_QUADRUPOLE
#define TREEPM_BLOCK_SIZE 32
#define TREEPM_OVERSUBSCRIPTION 4

#define SINK_BUCKET_SIZE 64
#define SOURCE_BUCKET_SIZE 64

#define MAX_RUNG 32
#define NINTERP 6
#define CHAIN_BW 2
#define NCELLS ((2*CHAIN_BW+1)*(2*CHAIN_BW+1)*(2*CHAIN_BW+1))
#define KICK_PP_MAX (37*32)
#define KICK_PC_MAX (4*32)

#define NCHILD 2
#define PHI_BW 4
#define NDIM 3

#define XDIM 0
#define YDIM 1
#define ZDIM 2

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
	fflush (stdout);
#endif
}

#include <tigerpm/containers.hpp>

namespace constants {
constexpr double mpc_to_cm = 3.086e+24;
constexpr double c = 2.99792458e10;
constexpr double H0 = 1e7 / mpc_to_cm;
constexpr double G = 6.67384e-8;
constexpr double sigma = 5.67051e-5;
}

#define FREAD(a,b,c,d) __safe_fread(a,b,c,d,__LINE__,__FILE__)

static void __safe_fread(void* src, size_t size, size_t count, FILE* fp, int line, const char* file) {
	auto read = fread(src, size, count, fp);
	if (read != count) {
		PRINT("Attempt to read %li elements of size %li in %s on line %i failed - only %li elements read.\n", count, size,
				file, line, read);
		abort();
	}
}



#endif /* TIGERFMM_HPP_ */
