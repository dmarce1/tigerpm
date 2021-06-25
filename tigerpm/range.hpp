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

inline std::array<int, NDIM> shift_up(std::array<int, NDIM> i) {
	std::array<int, NDIM> j;
	j[2] = i[0];
	j[0] = i[1];
	j[1] = i[2];
	return j;
}

inline std::array<int, NDIM> shift_down(std::array<int, NDIM> i) {
	std::array<int, NDIM> j;
	j[1] = i[0];
	j[2] = i[1];
	j[0] = i[2];
	return j;
}

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

	range() = default;
	range(const range&) = default;
	range(range&&) = default;
	range& operator=(const range&) = default;
	range& operator=(range&&) = default;

	inline range shift(const std::array<T, NDIM>& s) const {
		range r = *this;
		for (int dim = 0; dim < NDIM; dim++) {
			r.begin[dim] += s[dim];
			r.end[dim] += s[dim];
		}
		return r;
	}

	range(const T& sz) {
		for (int dim = 0; dim < NDIM; dim++) {
			begin[dim] = T(0);
			end[dim] = sz;
		}
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

	inline bool contains(const std::array<T, NDIM>& p) const {
		for (int dim = 0; dim < NDIM; dim++) {
			if (p[dim] < begin[dim] || p[dim] >= end[dim]) {
				return false;
			}
		}
		return true;
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

	inline int longest_dim() const {
		int max_dim;
		T max_span = 0;
		for (int dim = 0; dim < NDIM; dim++) {
			const T span = end[dim] - begin[dim];
			if (span > max_span) {
				max_span = span;
				max_dim = dim;
			}
		}
		return max_dim;
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

	range<T> shift_up() const {
		range<T> rc;
		rc.begin = ::shift_up(begin);
		rc.end = ::shift_up(end);
		return rc;
	}

	range<T> shift_down() const {
		range<T> rc;
		rc.begin = ::shift_down(begin);
		rc.end = ::shift_down(end);
		return rc;
	}

	inline int index(int xi, int yi, int zi) const {
		const auto spanz = end[2] - begin[2];
		const auto spany = end[1] - begin[1];
		return spanz * (spany * (xi - begin[0]) + (yi - begin[1])) + (zi - begin[2]);
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

	inline range<T> pad(T dx = T(1)) const {
		range<T> r;
		for (int dim = 0; dim < NDIM; dim++) {
			r.begin[dim] = begin[dim] - dx;
			r.end[dim] = end[dim] + dx;
		}
		return r;
	}

};

#endif /* RANGE_HPP_ */
