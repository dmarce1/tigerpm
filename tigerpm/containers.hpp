/*
 * containers.hpp
 *
 *  Created on: Jun 30, 2021
 *      Author: dmarce1
 */

#ifndef CONTAINERS_HPP_
#define CONTAINERS_HPP_

#include <vector>
#include <array>

#define DO_BOUNDS_CHECK(i) \
		if( i < 0 || i >= this->size() ) { \
			PRINT( "Bounds error in %s on line %i - %i is not between 0 and %i\n", __FILE__, __LINE__, i, this->size()); \
			abort(); \
		}

template<class T>
class vector: public std::vector<T> {
	using base_type = std::vector<T>;
	using base_type::base_type;

public:
#ifdef CHECK_BOUNDS
	inline T& operator[](int i ) {
		DO_BOUNDS_CHECK(i);
		return std::vector<T>::operator[](i);
	}
	inline const T operator[](int i ) const {
		DO_BOUNDS_CHECK(i);
		return std::vector<T>::operator[](i);
	}
#endif
};

template<class T, int N>
class array: public std::array<T, N> {
	using base_type = std::array<T, N>;
	using base_type::base_type;
public:
#ifdef CHECK_BOUNDS
	inline T& operator[](int i ) {
		DO_BOUNDS_CHECK(i);
		return std::array<T,N>::operator[](i);
	}
	inline const T operator[](int i ) const {
		DO_BOUNDS_CHECK(i);
		return std::array<T,N>::operator[](i);
	}
#endif

};

#endif /* CONTAINERS_HPP_ */
