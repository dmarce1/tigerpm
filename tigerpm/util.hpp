#pragma once

#include <tigerpm/cuda.hpp>
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
