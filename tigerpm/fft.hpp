/*
 * fft.hpp
 *
 *  Created on: Jun 23, 2021
 *      Author: dmarce1
 */

#ifndef FFT_HPP_
#define FFT_HPP_


#include <tigerpm/tigerpm.hpp>
#include <tigerpm/range.hpp>

void fft3d_init(int N);
void fft3d_execute();
void fft3d_accumulate(const range<int>&, const std::vector<float>&);
void fft3d_destroy();

#endif /* FFT_HPP_ */
