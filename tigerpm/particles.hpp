/*
 * particles.hpp
 *
 *  Created on: Jun 25, 2021
 *      Author: dmarce1
 */

#ifndef PARTICLES_HPP_
#define PARTICLES_HPP_

#include <tigerpm/fixed.hpp>
#include <tigerpm/range.hpp>

#ifndef PARTICLES_CPP
extern std::array<std::vector<fixed32>, NDIM> particles_X;
extern std::array<std::vector<float>, NDIM> particles_U;
#endif

struct particle {
	std::array<fixed32, NDIM> x;
	std::array<float, NDIM> v;
	template<class A>
	void serialize(A&& arc, unsigned) {
		arc & x;
		arc & v;
	}
};

int particles_size();
void particles_resize(size_t new_size);
void particles_domain_sort();
void particles_random_init();
range<int> particles_get_local_box();
std::array<int, NDIM> particles_mesh_loc(int index);


inline fixed32& particles_pos(int dim, int index) {
	return particles_X[dim][index];
}

inline float& particles_vel(int dim, int index) {
	return particles_U[dim][index];
}

inline particle particles_get_particle(int index) {
	particle p;
	for (int dim = 0; dim < NDIM; dim++) {
		p.x[dim] = particles_pos(dim, index);
		p.v[dim] = particles_vel(dim, index);
	}
	return p;
}

inline void particles_set_particle(const particle& p, int index) {
	for (int dim = 0; dim < NDIM; dim++) {
		particles_pos(dim, index) = p.x[dim];
		particles_vel(dim, index) = p.v[dim];
	}
}

#endif /* PARTICLES_HPP_ */
