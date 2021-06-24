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
#include <tigerpm/complex.hpp>

void fft3d_init(int N);
void fft3d_execute();
std::vector<float> fft3d_read_real(const range<int>&);
std::vector<cmplx> fft3d_read_complex(const range<int>&);
void fft3d_accumulate(const range<int>&, const std::vector<float>&);
void fft3d_destroy();

#endif /* FFT_HPP_ */
