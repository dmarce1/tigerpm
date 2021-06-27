/*
 * gravity_short.hpp
 *
 *  Created on: Jun 26, 2021
 *      Author: dmarce1
 */

#ifndef GRAVITY_SHORT_HPP_
#define GRAVITY_SHORT_HPP_

#include <tigerpm/fixed.hpp>
#include <tigerpm/tigerpm.hpp>

#include <array>
#include <vector>

#define EWALD_BLOCK_SIZE 128


CUDA_EXPORT inline float distance(fixed32 a, fixed32 b) {
	return (fixed<int32_t>(a) - fixed<int32_t>(b)).to_float();
}


std::pair<std::vector<double>, std::array<std::vector<double>, NDIM>> gravity_short_ewald_call_kernel(
		const std::vector<fixed32>& sinkx, const std::vector<fixed32>& sinky, const std::vector<fixed32>& sinkz);

void gravity_short_ewald_compare(int Nsamples);

#endif /* GRAVITY_SHORT_HPP_ */
