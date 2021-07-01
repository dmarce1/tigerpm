/*
 * gravity_long.hpp
 *
 *  Created on: Jun 25, 2021
 *      Author: dmarce1
 */

#ifndef GRAVITY_LONG_HPP_
#define GRAVITY_LONG_HPP_

#include <tigerpm/tigerpm.hpp>
#include <tigerpm/range.hpp>

#include <array>
#include <utility>

using gravity_long_type = int;
#define GRAVITY_LONG_PM 0
#define GRAVITY_LONG_PME 1

void gravity_long_compute(gravity_long_type type = GRAVITY_LONG_PM);
vector<float> gravity_long_get_phi(const range<int>&);
range<int> gravity_long_get_phi_box();
std::pair<float, array<float, NDIM>> gravity_long_force_at(const array<double, NDIM>& pos);
std::pair<float, array<float, NDIM>> gravity_long_force_at(const array<double, NDIM>& pos);

#endif /* GRAVITY_LONG_HPP_ */
