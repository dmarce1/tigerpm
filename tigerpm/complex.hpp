/*
 * complex.hpp
 *
 *  Created on: Jun 20, 2021
 *      Author: dmarce1
 */

#ifndef COMPLEX_HPP_
#define COMPLEX_HPP_

#include <algorithm>

template<class T = float>
class complex {
	T x, y;
public:
	complex() = default;

	complex(float a) {
		x = a;
		y = T(0.0);
	}

	complex(float a, float b) {
		x = a;
		y = b;
	}

	complex& operator+=(complex other) {
		x += other.x;
		y += other.y;
		return *this;
	}

	complex& operator-=(complex other) {
		x -= other.x;
		y -= other.y;
		return *this;
	}

	complex operator*(complex other) const {
		complex a;
		a.x = x * other.x - y * other.y;
		a.y = x * other.y + y * other.x;
		return a;
	}

	complex operator/(complex other) const {
		return *this * other.conj() / other.norm();
	}

	complex operator/(float other) const {
		complex b;
		b.x = x / other;
		b.y = y / other;
		return b;
	}

	complex operator*(float other) const {
		complex b;
		b.x = x * other;
		b.y = y * other;
		return b;
	}

	complex operator+(complex other) const {
		complex a;
		a.x = x + other.x;
		a.y = y + other.y;
		return a;
	}

	complex operator-(complex other) const {
		complex a;
		a.x = x - other.x;
		a.y = y - other.y;
		return a;
	}

	complex conj() const {
		complex a;
		a.x = x;
		a.y = -y;
		return a;
	}

	float real() const {
		return x;
	}

	float imag() const {
		return y;
	}

	float& real() {
		return x;
	}

	float& imag() {
		return y;
	}

	float norm() const {
		return ((*this) * conj()).real();
	}

	float abs() const {
		return sqrtf(norm());
	}

	complex operator-() const {
		complex a;
		a.x = -x;
		a.y = -y;
		return a;
	}
	template<class A>
	void serialize(A&& arc, unsigned) {
		arc & x;
		arc & y;
	}
};


template<class T>
inline void swap(complex<T>& a, complex<T>& b) {
	std::swap(a.real(), b.real());
	std::swap(a.imag(), b.imag());
}

using cmplx = complex<float>;

#endif /* COMPLEX_HPP_ */
