/*
 * gravity_long.hpp
 *
 *  Created on: Jun 25, 2021
 *      Author: dmarce1
 */

#ifndef GRAVITY_LONG_HPP_
#define GRAVITY_LONG_HPP_




void gravity_long_compute();
std::pair<float, std::array<float, NDIM>> gravity_long_force_at(const std::array<double, NDIM>& pos);

#endif /* GRAVITY_LONG_HPP_ */
