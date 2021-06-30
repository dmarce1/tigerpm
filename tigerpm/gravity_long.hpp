/*
 * gravity_long.hpp
 *
 *  Created on: Jun 25, 2021
 *      Author: dmarce1
 */

#ifndef GRAVITY_LONG_HPP_
#define GRAVITY_LONG_HPP_


#include <tigerpm/tigerpm.hpp>

#include <array>
#include <utility>

void gravity_long_compute();
std::pair<float, array<float, NDIM>> gravity_long_force_at(const array<double, NDIM>& pos);
std::pair<float, array<float, NDIM>> gravity_long_force_at(const array<double, NDIM>& pos);

#endif /* GRAVITY_LONG_HPP_ */
