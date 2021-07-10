#pragma once

#include <tigerpm/util.hpp>

#define WARP_SIZE 32
#define MULTIPOLE_SIZE 4
#define EXPANSION_SIZE 10

using expansion = array<float, EXPANSION_SIZE>;
using multipole = array<float, MULTIPOLE_SIZE>;

#ifdef __CUDACC__
__device__ inline void greens_function(array<float,10> &D, float dx, float dy, float dz, float inv2rs) {
	const float twooversqrtpi = 1.12837917e+00;
	const float r2 = sqr(dx, dy, dz);
	const float rinv = rsqrtf(r2);
	const float r = r2 * rinv;
	const float r0 = r * inv2rs;
	float exp0;
	const float erfc0 = erfcexp(r0, &exp0);
	const float r02 = r0 * r0;
	const float c0 = -2.f * r * inv2rs;
	const float d0 = -erfc0 * rinv;
	float e0 = twooversqrtpi * exp0 * rinv * inv2rs;
	const float d1 = fmaf(float(-1) * d0, rinv, e0);
	e0 *= c0;
	const float d2 = fmaf(float(-3) * d1, rinv, e0);
	e0 *= c0;
	const float rinv0 = 1.f;
	const float rinv1 = rinv;
	const float Drinvpow_0_0 = d0 * rinv0;
	const float Drinvpow_1_0 = d1 * rinv0;
	const float Drinvpow_1_1 = d1 * rinv1;
	const float Drinvpow_2_0 = d2 * rinv0;
	array<float,NDIM> dxrinv;
	dxrinv[0] = dx * rinv;
	dxrinv[1] = dy * rinv;
	dxrinv[2] = dz * rinv;
	const float x000 = float(1);
	const float& x100 = dxrinv[0];
	const float& x010 = dxrinv[1];
	const float& x001 = dxrinv[2];
	const float x002 = x001 * x001;
	const float x011 = x010 * x001;
	const float x020 = x010 * x010;
	const float x101 = x100 * x001;
	const float x110 = x100 * x010;
	const float x200 = x100 * x100;
	float x_2_1_000 = x002;
	x_2_1_000 += x020;
	x_2_1_000 += x200;
	x_2_1_000 *= Drinvpow_1_1;
	D[0] = fmaf(x000, Drinvpow_0_0, D[0]);
	D[7] += x_2_1_000;
	D[3] = fmaf(x001, Drinvpow_1_0, D[3]);
	D[1] = fmaf(x100, Drinvpow_1_0, D[1]);
	D[9] = fmaf(x002, Drinvpow_2_0, D[9]);
	D[6] = fmaf(x101, Drinvpow_2_0, D[6]);
	D[9] += x_2_1_000;
	D[5] = fmaf(x110, Drinvpow_2_0, D[5]);
	D[2] = fmaf(x010, Drinvpow_1_0, D[2]);
	D[4] = fmaf(x200, Drinvpow_2_0, D[4]);
	D[8] = fmaf(x011, Drinvpow_2_0, D[8]);
	D[4] += x_2_1_000;
	D[7] = fmaf(x020, Drinvpow_2_0, D[7]);
}

#endif
__device__
inline int M2L_kernel(array<float, 4>& L, const array<float, 4>& M, const array<float, 10>& D, bool do_phi) {
	if( do_phi ) {
		L[0] = fmaf(M[0], D[0], L[0]);
		L[0] = fmaf(M[3], D[3], L[0]);
		L[0] = fmaf(M[2], D[2], L[0]);
		L[0] = fmaf(M[1], D[1], L[0]);
	}
	L[1] = fmaf(M[0], D[1], L[1]);
	L[2] = fmaf(M[2], D[7], L[2]);
	L[1] = fmaf(M[3], D[6], L[1]);
	L[2] = fmaf(M[1], D[5], L[2]);
	L[1] = fmaf(M[2], D[5], L[1]);
	L[3] = fmaf(M[0], D[3], L[3]);
	L[1] = fmaf(M[1], D[4], L[1]);
	L[3] = fmaf(M[3], D[9], L[3]);
	L[2] = fmaf(M[0], D[2], L[2]);
	L[3] = fmaf(M[2], D[8], L[3]);
	L[2] = fmaf(M[3], D[8], L[2]);
	L[3] = fmaf(M[1], D[6], L[3]);
	return 24 + do_phi * 8;
}

__device__
inline int M2L_kernel(array<float, 10>& L, const array<float, 4>& M, const array<float, 10>& D, bool do_phi) {
	if( do_phi ) {
		L[0] = fmaf(M[0], D[0], L[0]);
		L[0] = fmaf(M[3], D[3], L[0]);
		L[0] = fmaf(M[2], D[2], L[0]);
		L[0] = fmaf(M[1], D[1], L[0]);
	}
	L[1] = fmaf(M[1], D[4], L[1]);
	L[3] = fmaf(M[2], D[8], L[3]);
	L[1] = fmaf(M[2], D[5], L[1]);
	L[3] = fmaf(M[1], D[6], L[3]);
	L[1] = fmaf(M[0], D[1], L[1]);
	L[3] = fmaf(M[0], D[3], L[3]);
	L[1] = fmaf(M[3], D[6], L[1]);
	L[4] = fmaf(M[0], D[4], L[4]);
	L[2] = fmaf(M[0], D[2], L[2]);
	L[5] = fmaf(M[0], D[5], L[5]);
	L[2] = fmaf(M[3], D[8], L[2]);
	L[6] = fmaf(M[0], D[6], L[6]);
	L[2] = fmaf(M[2], D[7], L[2]);
	L[7] = fmaf(M[0], D[7], L[7]);
	L[2] = fmaf(M[1], D[5], L[2]);
	L[8] = fmaf(M[0], D[8], L[8]);
	L[3] = fmaf(M[3], D[9], L[3]);
	L[9] = fmaf(M[0], D[9], L[9]);
	return 36 + do_phi * 8;
}

inline array<float, 4> P2M_kernel(array<float, NDIM>& X) {
	array<float, 4> M;
	X[0] = -X[0];
	X[1] = -X[1];
	X[2] = -X[2];
	M[0] = float(1);
	M[1] = X[0];
	M[2] = X[1];
	M[3] = X[2];
	return M;
/* FLOPS = 3*/
}

inline array<float, 4> M2M_kernel(const array<float,4>& Ma, array<float, NDIM>& X) {
	array<float, 4> Mb;
	X[0] = -X[0];
	X[1] = -X[1];
	X[2] = -X[2];
	const float x000 = float(1);
	const float& x100 = X[0];
	const float& x010 = X[1];
	const float& x001 = X[2];
	Mb[0] = Ma[0];
	Mb[1] = Ma[1];
	Mb[2] = Ma[2];
	Mb[3] = Ma[3];
	Mb[1] = fmaf( x100, Ma[0], Mb[1]);
	Mb[3] = fmaf( x001, Ma[0], Mb[3]);
	Mb[2] = fmaf( x010, Ma[0], Mb[2]);
	return Mb;
/* FLOPS = 6*/
}

static __device__ char Ldest1[5] = { 1,1,1,2,2};
static __device__ float factor1[5] = { 1.00000000e+00,1.00000000e+00,1.00000000e+00,1.00000000e+00,1.00000000e+00};
static __device__ char xsrc1[5] = { 1,2,3,1,2};
static __device__ char Lsrc1[5] = { 4,5,6,5,7};
static __device__ char Ldest2[4] = { 2,3,3,3};
static __device__ float factor2[4] = { 1.00000000e+00,1.00000000e+00,1.00000000e+00,1.00000000e+00};
static __device__ char xsrc2[4] = { 3,1,2,3};
static __device__ char Lsrc2[4] = { 8,6,8,9};
static __device__ float phi_factor[9] = { 1.00000000e+00,5.00000000e-01,1.00000000e+00,1.00000000e+00,5.00000000e-01,1.00000000e+00,1.00000000e+00,1.00000000e+00,5.00000000e-01};
static __device__ char phi_Lsrc[9] = { 3,9,2,8,7,1,6,5,4};

#ifdef __CUDACC__
__device__
inline array<float, 10> L2L_kernel(const array<float, 10>& La, const array<float, NDIM>& X, bool do_phi) {
	const int tid = threadIdx.x;
	array<float, 10> Lb;
	for( int i = 0; i < EXPANSION_SIZE; i++) {
		Lb[i] = 0.0f;
	}
	for( int i = tid; i < EXPANSION_SIZE; i += WARP_SIZE ) {
		Lb[i] = La[i];
	}
	array<float,10> dx;
	dx[0] = float(1);
	dx[1] = X[0];
	dx[2] = X[1];
	dx[3] = X[2];
	dx[9]= dx[3] * dx[3];
	dx[8]= dx[2] * dx[3];
	dx[7]= dx[2] * dx[2];
	dx[6]= dx[1] * dx[3];
	dx[5]= dx[1] * dx[2];
	dx[4]= dx[1] * dx[1];
	for( int i = tid; i < 4; i+=WARP_SIZE) {
		Lb[Ldest1[i]] = fmaf(factor1[i] * dx[xsrc1[i]], La[Lsrc1[i]], Lb[Ldest1[i]]);
		Lb[Ldest2[i]] = fmaf(factor2[i] * dx[xsrc2[i]], La[Lsrc2[i]], Lb[Ldest2[i]]);
	}
	Lb[Ldest1[4]] = fmaf(factor1[4] * dx[xsrc1[4]], La[Lsrc1[4]], Lb[Ldest1[4]]);
	if( do_phi ) {
		for( int i = tid; i < 9; i+=WARP_SIZE) {
			Lb[0] = fmaf(phi_factor[i] * dx[phi_Lsrc[i]], La[phi_Lsrc[i]], Lb[0]);
		}
	}
	for (int P = warpSize / 2; P >= 1; P /= 2) {
		for (int i = 0; i < EXPANSION_SIZE; i++) {
			Lb[i] += __shfl_xor_sync(0xffffffff, Lb[i], P);
		}
	}
	return Lb;
/* FLOPS = 42 + do_phi * 36*/
}
#endif

__device__
inline array<float, 4> L2P_kernel(const array<float, 10>& La, const array<float, NDIM>& X, bool do_phi) {
	array<float, 4> Lb;
//	const float x000 = float(1);
	const float& x100 = X[0];
	const float& x010 = X[1];
	const float& x001 = X[2];
	const float x002 = x001 * x001;
	const float x011 = x010 * x001;
	const float x020 = x010 * x010;
	const float x101 = x100 * x001;
	const float x110 = x100 * x010;
	const float x200 = x100 * x100;
	Lb[0] = La[0];
	Lb[1] = La[1];
	Lb[2] = La[2];
	Lb[3] = La[3];
	if( do_phi ) {
		Lb[0] = fmaf( x001, La[3], Lb[0]);
		Lb[0] = fmaf(float(5.00000000e-01) * x002, La[9], Lb[0]);
		Lb[0] = fmaf( x010, La[2], Lb[0]);
		Lb[0] = fmaf( x011, La[8], Lb[0]);
		Lb[0] = fmaf(float(5.00000000e-01) * x020, La[7], Lb[0]);
		Lb[0] = fmaf( x100, La[1], Lb[0]);
		Lb[0] = fmaf( x101, La[6], Lb[0]);
		Lb[0] = fmaf( x110, La[5], Lb[0]);
		Lb[0] = fmaf(float(5.00000000e-01) * x200, La[4], Lb[0]);
	}
	Lb[1] = fmaf( x100, La[4], Lb[1]);
	Lb[2] = fmaf( x001, La[8], Lb[2]);
	Lb[1] = fmaf( x010, La[5], Lb[1]);
	Lb[3] = fmaf( x100, La[6], Lb[3]);
	Lb[1] = fmaf( x001, La[6], Lb[1]);
	Lb[3] = fmaf( x010, La[8], Lb[3]);
	Lb[2] = fmaf( x100, La[5], Lb[2]);
	Lb[3] = fmaf( x001, La[9], Lb[3]);
	Lb[2] = fmaf( x010, La[7], Lb[2]);
	return Lb;
/* FLOPS = 45*/
}

