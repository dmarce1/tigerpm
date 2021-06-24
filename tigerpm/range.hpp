/*
 * range.hpp
 *
 *  Created on: Jun 23, 2021
 *      Author: dmarce1
 */

#ifndef RANGE_HPP_
#define RANGE_HPP_

#include <tigerpm/tigerpm.hpp>
#include <array>

template<class T>
struct range {
	std::array<T, NDIM> begin;
	std::array<T, NDIM> end;

	inline range intersection(const range& other) const {
		range I;
		for (int dim = 0; dim < NDIM; dim++) {
			I.begin[dim] = std::max(begin[dim], other.begin[dim]);
			I.end[dim] = std::min(end[dim], other.end[dim]);
		}
		return I;
	}

	inline bool empty() const {
		return volume() == T(0);
	}

	inline bool contains(const range<T>& box) const {
		bool rc = true;
		for (int dim = 0; dim < NDIM; dim++) {
			if (begin[dim] > box.begin[dim]) {
				rc = false;
				break;
			}
			if (end[dim] < box.end[dim]) {
				rc = false;
				break;
			}
		}
		return rc;
	}

	inline std::string to_string() const {
		std::string str;
		for (int dim = 0; dim < NDIM; dim++) {
			str += std::to_string(dim) + ":(";
			str += std::to_string(begin[dim]) + ",";
			str += std::to_string(end[dim]) + ") ";
		}
		return str;
	}

	inline std::pair<range<T>, range<T>> split() const {
		auto left = *this;
		auto right = *this;
		int max_dim;
		T max_span = 0;
		for (int dim = 0; dim < NDIM; dim++) {
			const T span = end[dim] - begin[dim];
			if (span > max_span) {
				max_span = span;
				max_dim = dim;
			}
		}
		const T mid = (end[max_dim] + begin[max_dim]) / T(2);
		left.end[max_dim] = right.begin[max_dim] = mid;
		return std::make_pair(left, right);
	}

	inline int index(std::array<T, NDIM> & i) const {
		const auto spanz = end[2] - begin[2];
		const auto spany = end[1] - begin[1];
		return spanz * (spany * (i[0] - begin[0]) + (i[1] - begin[1])) + (i[2] - begin[2]);
	}

	inline range<int> transpose(int dim1, int dim2) const {
		auto rc = *this;
		std::swap(rc.begin[dim1], rc.begin[dim2]);
		std::swap(rc.end[dim1], rc.end[dim2]);
		return rc;
	}

	inline T volume() const {
		T vol = T(1);
		for (int dim = 0; dim < NDIM; dim++) {
			vol *= end[dim] - begin[dim];
		}
		return vol < T(0) ? T(0) : vol;
	}

	template<class A>
	void serialize(A&& arc, unsigned) {
		for (int dim = 0; dim < NDIM; dim++) {
			arc & begin[dim];
			arc & end[dim];
		}
	}

};

#endif /* RANGE_HPP_ */
