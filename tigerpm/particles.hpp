/*
 * particles.hpp
 *
 *  Created on: Jun 25, 2021
 *      Author: dmarce1
 */

#ifndef PARTICLES_HPP_
#define PARTICLES_HPP_

#include <tigerpm/fixed.hpp>
#include <tigerpm/options.hpp>
#include <tigerpm/range.hpp>

#include <vector>

#ifndef PARTICLES_CPP
extern array<vector<fixed32>, NDIM> particles_X;
extern array<vector<float>, NDIM> particles_U;
extern vector<char> particles_R;
#ifdef FORCE_TEST
extern vector<float> particles_P;
extern array<vector<float>, NDIM> particles_G;
#endif
#endif

struct particle {
	array<fixed32, NDIM> x;
	array<float, NDIM> v;
	char r;
#ifdef FORCE_TEST
	array<float, NDIM> g;
	float p;
#endif
	template<class A>
	void serialize(A&& arc, unsigned) {
		arc & x;
		arc & v;
		arc & r;
#ifdef FORCE_TEST
		arc & g;
		arc & p;
#endif
	}
};

struct particle_pos {
	array<fixed32, NDIM> x;
	template<class A>
	void serialize(A&& arc, unsigned) {
		arc & x;
	}
};

int particles_size();
void particles_resize(size_t new_size);
void particles_resize_pos(size_t new_size);
void particles_random_init();
void particles_domain_sort();
range<int> particles_get_local_box();
int particles_local_size();
vector<int> particles_per_rank();
vector<particle> particles_sample(const vector<int>&);
vector<particle> particles_sample(int);
void particles_sphere_init(float radius);
vector<int> particles_mesh_count();
int particles_sort(int begin, int end, double xmid, int xdim);
void particles_load(FILE* fp);
void particles_save(FILE* fp);

inline array<int, NDIM> particles_mesh_loc(int index) {
	static const double N = get_options().chain_dim;
	array<int, NDIM> i;
	for (int dim = 0; dim < NDIM; dim++) {
		i[dim] = particles_X[dim][index].to_double() * N;
	}
	return i;
}

inline fixed32& particles_pos(int dim, int index) {
	return particles_X[dim][index];
}

inline float& particles_vel(int dim, int index) {
	return particles_U[dim][index];
}

inline char& particles_rung(int index) {
	return particles_R[index];
}

#ifdef FORCE_TEST
inline float& particles_pot(int index) {
	return particles_P[index];
}

inline float& particles_gforce(int dim, int index) {
	return particles_G[dim][index];
}
#endif

inline particle particles_get_particle(int index) {
	particle p;
	for (int dim = 0; dim < NDIM; dim++) {
		p.x[dim] = particles_pos(dim, index);
		p.v[dim] = particles_vel(dim, index);
#ifdef FORCE_TEST
		p.g[dim] = particles_gforce(dim, index);
#endif
	}
	p.r = particles_rung(index);
#ifdef FORCE_TEST
	p.p = particles_pot(index);
#endif

	return p;
}

inline particle_pos particles_get_particle_pos(int index) {
	particle_pos p;
	for (int dim = 0; dim < NDIM; dim++) {
		p.x[dim] = particles_pos(dim, index);
	}
	return p;
}

inline void particles_set_particle(const particle& p, int index) {
	for (int dim = 0; dim < NDIM; dim++) {
		particles_pos(dim, index) = p.x[dim];
		particles_vel(dim, index) = p.v[dim];
#ifdef FORCE_TEST
		particles_gforce(dim, index) = p.g[dim];
#endif
	}
	particles_rung(index) = p.r;
#ifdef FORCE_TEST
	particles_pot(index) = p.p;
#endif
}

inline void particles_set_particle_pos(const particle_pos& p, int index) {
	for (int dim = 0; dim < NDIM; dim++) {
		particles_pos(dim, index) = p.x[dim];
	}
}

#endif /* PARTICLES_HPP_ */
