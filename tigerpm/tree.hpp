/*
 * tree.hpp
 *
 *  Created on: Jul 2, 2021
 *      Author: dmarce1
 */

#ifndef TREE_HPP_
#define TREE_HPP_

#include <tigerpm/tigerpm.hpp>
#include <tigerpm/cuda.hpp>
#include <tigerpm/fixed.hpp>
#include <tigerpm/chainmesh.hpp>

struct quadrupole {
	float xx;
	float xy;
	float xz;
	float yy;
	float yz;
	float zz;
};

struct tree_node {
	array<fixed32, NDIM> x;
	float mass;
	float radius;
	array<int, NCHILD> children;
	int pbegin;
	int pend;
#ifdef USE_QUADRUPOLE
	quadrupole q;
#endif
};

struct sink_bucket {
	int src_begin;
	int src_end;
	int snk_begin;
	int snk_end;
	float radius;
	array<fixed32,NDIM> x;
};

class tree {
public:
	tree_node* nodes;
	int sz;
	int cap;
	bool device;
	void resize(int new_size);
	size_t size() const {
		return sz;
	}
	tree();
	~tree();
	tree& operator=(const tree& other);
	tree & operator=(tree && other);
	tree(const tree& other);
	tree(tree && other);
	tree to_device() const;
	int allocate();
	CUDA_EXPORT inline
	fixed32 get_x(int dim, int i) const {
		return nodes[i].x[dim];
	}
	CUDA_EXPORT inline int get_pbegin(int i) const {
		return nodes[i].pbegin;
	}
	CUDA_EXPORT inline int get_pend(int i) const {
		return nodes[i].pend;
	}
#ifdef USE_QUADRUPOLE
	CUDA_EXPORT inline quadrupole get_quadrupole(int i) const {
		return nodes[i].q;
	}
#endif
	CUDA_EXPORT inline
	float get_mass(int i) const {
		return nodes[i].mass;
	}
	CUDA_EXPORT inline
	float is_leaf(int i) const {
		return nodes[i].children[0] == -1;
	}
	CUDA_EXPORT inline
	float get_radius(int i) const {
		return nodes[i].radius;
	}
	CUDA_EXPORT inline
	array<int, NCHILD> get_children(int i) const {
		return nodes[i].children;
	}
	inline void set(tree_node node, int i) {
		nodes[i] = node;
	}
	inline void adjust_indexes(int dif) {
		for( int i = 0; i < sz; i++) {
			nodes[i].pbegin += dif;
			nodes[i].pend += dif;
		}
	}
}
;

std::pair<tree, vector<sink_bucket>> tree_create(const array<int, NDIM>& cell_index, chaincell cell);

#endif /* TREE_HPP_ */
