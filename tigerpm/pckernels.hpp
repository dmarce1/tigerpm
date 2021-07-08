/*
 * pckernels.hpp
 *
 *  Created on: Jul 7, 2021
 *      Author: dmarce1
 */

#ifndef PCKERNELS_HPP_
#define PCKERNELS_HPP_

#include <tigerpm/tigerpm.hpp>
#include <tigerpm/util.hpp>

#define MULTIPOLE_SIZE 10
#define GREENS_SIZE 20

using multipole = array<float,MULTIPOLE_SIZE>;
using greens = array<float,GREENS_SIZE>;

__device__
inline int pc_interaction(array<float, NDIM+1>& L, const array<float, 10>& M, const array<float, 20>& D, bool do_phi) {
	if (do_phi) {
		L[0] = M[0] * D[0];
		L[0] = fmaf(M[3], D[3], L[0]);
		L[0] = fmaf(float(5.00000000e-01) * M[9], D[9], L[0]);
		L[0] = fmaf(M[2], D[2], L[0]);
		L[0] = fmaf(M[8], D[8], L[0]);
		L[0] = fmaf(float(5.00000000e-01) * M[7], D[7], L[0]);
		L[0] = fmaf(M[1], D[1], L[0]);
		L[0] = fmaf(M[6], D[6], L[0]);
		L[0] = fmaf(M[5], D[5], L[0]);
		L[0] = fmaf(float(5.00000000e-01) * M[4], D[4], L[0]);
	}
	L[1] = float(5.00000000e-01) * M[4] * D[10];
	L[2] = M[8] * D[17];
	L[1] = fmaf(M[5], D[11], L[1]);
	L[2] = fmaf(M[2], D[7], L[2]);
	L[1] = fmaf(M[6], D[12], L[1]);
	L[2] = fmaf(float(5.00000000e-01) * M[9], D[18], L[2]);
	L[1] = fmaf(M[1], D[4], L[1]);
	L[2] = fmaf(M[3], D[8], L[2]);
	L[1] = fmaf(float(5.00000000e-01) * M[7], D[13], L[1]);
	L[2] = fmaf(M[0], D[2], L[2]);
	L[1] = fmaf(M[8], D[14], L[1]);
	L[3] = M[0] * D[3];
	L[1] = fmaf(M[2], D[5], L[1]);
	L[3] = fmaf(float(5.00000000e-01) * M[4], D[12], L[3]);
	L[1] = fmaf(float(5.00000000e-01) * M[9], D[15], L[1]);
	L[3] = fmaf(M[5], D[14], L[3]);
	L[1] = fmaf(M[3], D[6], L[1]);
	L[3] = fmaf(M[6], D[15], L[3]);
	L[1] = fmaf(M[0], D[1], L[1]);
	L[3] = fmaf(M[1], D[6], L[3]);
	L[2] = fmaf(float(5.00000000e-01) * M[7], D[16], L[2]);
	L[3] = fmaf(float(5.00000000e-01) * M[7], D[17], L[3]);
	L[2] = fmaf(float(5.00000000e-01) * M[4], D[11], L[2]);
	L[3] = fmaf(M[8], D[18], L[3]);
	L[2] = fmaf(M[5], D[13], L[2]);
	L[3] = fmaf(M[2], D[8], L[3]);
	L[2] = fmaf(M[6], D[14], L[2]);
	L[3] = fmaf(float(5.00000000e-01) * M[9], D[19], L[3]);
	L[2] = fmaf(M[1], D[5], L[2]);
	L[3] = fmaf(M[3], D[9], L[3]);
	return 69 + do_phi * 23;
}

inline array<float, 10> monopole_translate(array<float, NDIM>& X) {
	array<float, 10> M;
	X[0] = -X[0];
	X[1] = -X[1];
	X[2] = -X[2];
	M[0] = float(1);
	M[1] = X[0];
	M[2] = X[1];
	M[3] = X[2];
	M[9] = M[3] * M[3];
	M[8] = M[2] * M[3];
	M[7] = M[2] * M[2];
	M[6] = M[1] * M[3];
	M[5] = M[1] * M[2];
	M[4] = M[1] * M[1];
	return M;
	/* FLOPS = 9*/
}

inline array<float, 10> multipole_translate(const array<float, 10>& Ma, array<float, NDIM>& X) {
	array<float, 10> Mb;
	X[0] = -X[0];
	X[1] = -X[1];
	X[2] = -X[2];
	const float& Mb000 = Mb[0];
	const float& Mb001 = Mb[3];
	const float& Mb002 = Mb[9];
	const float& Mb010 = Mb[2];
	const float& Mb011 = Mb[8];
	const float& Mb020 = Mb[7];
	const float& Mb100 = Mb[1];
	const float& Mb101 = Mb[6];
	const float& Mb110 = Mb[5];
	const float& Mb200 = Mb[4];
	const float& x100 = X[0];
	const float& x010 = X[1];
	const float& x001 = X[2];
	const float x002 = x001 * x001;
	const float x011 = x010 * x001;
	const float x020 = x010 * x010;
	const float x101 = x100 * x001;
	const float x110 = x100 * x010;
	const float x200 = x100 * x100;
	Mb[0] = Ma[0];
	Mb[1] = Ma[1];
	Mb[2] = Ma[2];
	Mb[3] = Ma[3];
	Mb[4] = Ma[4];
	Mb[5] = Ma[5];
	Mb[6] = Ma[6];
	Mb[7] = Ma[7];
	Mb[8] = Ma[8];
	Mb[9] = Ma[9];
	Mb[1] = fmaf(x100, Ma[0], Mb[1]);
	Mb[6] = fmaf(x100, Ma[3], Mb[6]);
	Mb[2] = fmaf(x010, Ma[0], Mb[2]);
	Mb[6] = fmaf(x001, Ma[1], Mb[6]);
	Mb[3] = fmaf(x001, Ma[0], Mb[3]);
	Mb[7] = fmaf(x020, Ma[0], Mb[7]);
	Mb[4] = fmaf(float(2.00000000e+00) * x100, Ma[1], Mb[4]);
	Mb[7] = fmaf(float(2.00000000e+00) * x010, Ma[2], Mb[7]);
	Mb[4] = fmaf(x200, Ma[0], Mb[4]);
	Mb[8] = fmaf(x011, Ma[0], Mb[8]);
	Mb[5] = fmaf(x110, Ma[0], Mb[5]);
	Mb[8] = fmaf(x010, Ma[3], Mb[8]);
	Mb[5] = fmaf(x100, Ma[2], Mb[5]);
	Mb[8] = fmaf(x001, Ma[2], Mb[8]);
	Mb[5] = fmaf(x010, Ma[1], Mb[5]);
	Mb[9] = fmaf(float(2.00000000e+00) * x001, Ma[3], Mb[9]);
	Mb[6] = fmaf(x101, Ma[0], Mb[6]);
	Mb[9] = fmaf(x002, Ma[0], Mb[9]);
	return Mb;
	/* FLOPS = 45*/
}

#endif /* PCKERNELS_HPP_ */
