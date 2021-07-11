#pragma once

#include <tigerpm/cuda.hpp>
#include <tigerpm/fixed.hpp>
#include <tigerpm/range.hpp>
#include <tigerpm/tigerpm.hpp>

range<int> find_my_box(int N);
void find_all_boxes(vector<range<int>>& boxes, int N);


template<class T>
CUDA_EXPORT inline T round_up(T num, T mod) {
	return num > 0 ? ((num - 1) / mod + 1) * mod : 0;
}

template<class T>
CUDA_EXPORT inline T sqr(T a) {
	return a * a;
}

template<class T>
CUDA_EXPORT inline T sqr(T a, T b, T c) {
	return fmaf(a, a, fmaf(b, b, sqr(c)));
}

__device__ inline float erfcexp(float x, float *e) {
	/*const float p(0.47047f);
	const float a1(0.3480242f);
	const float a2(-0.0958798f);
	const float a3(0.7478556f);
	const float t1 = 1.f / fmaf(p, x, 1.f);
	const float t2 = t1 * t1;
	const float t3 = t2 * t1;
	*e = expf(-x * x);
	return fmaf(a1, t1, fmaf(a2, t2, a3 * t3)) * *e;*/
	const float p(0.3275911f);
	 const float a1(0.254829592f);
	 const float a2(-0.284496736f);
	 const float a3(1.421413741f);
	 const float a4(-1.453152027f);
	 const float a5(1.061405429f);
	 const float t1 = 1.f / fmaf(p, x, 1.f);
	 const float t2 = t1 * t1;
	 const float t3 = t2 * t1;
	 const float t4 = t2 * t2;
	 const float t5 = t2 * t3;
	 *e = expf(-x * x);
	 return fmaf(a1, t1, fmaf(a2, t2, fmaf(a3, t3, fmaf(a4, t4, a5 * t5)))) * *e;
}

CUDA_EXPORT inline float load(float* number) {
#ifdef __CUDA_ARCH__
	return __ldg(number);
#else
	return *number;
#endif
}

CUDA_EXPORT inline int load(int* number) {
#ifdef __CUDA_ARCH__
	return __ldg(number);
#else
	return *number;
#endif
}

CUDA_EXPORT inline fixed32 load(fixed32* number) {
#ifdef __CUDA_ARCH__
	union u {
		fixed32 a;
		float b;
	};
	u c;
	c.b = __ldg((float*) number);
	return c.a;
#else
	return *number;
#endif
}

inline float sinc(float x) {
	if( x != 0.0 ) {
		return std::sin(x) / x;
	} else {
		return 1.0;
	}
}

CUDA_EXPORT
inline float tsc(float x) {
	const float absx = fabsf(x);
	if (absx < 0.5) {
		return 0.75 - sqr(x);
	} else if (absx < 1.5) {
		return 0.5 * sqr(1.5 - absx);
	} else {
		return 0.0;
	}
}

CUDA_EXPORT
inline float cloud4(float x) {
	float q1 = std::abs(x);
	float q2 = q1 * q1;
	float q3 = q2 * q1;
	if (q1 < 1.0) {
		return 0.16666667 * (4.0 - 6.0 * q2 + 3.0 * q3);
	} else if (q1 < 2.0) {
		return 0.16666667 * (8.0 - 12.0 * q1 + 6.0 * q2 - q3);
	} else {
		return 0.0;
	}
}
