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
#include <tigerpm/kernels.hpp>


struct tree_node {
	array<fixed32, NDIM> x;
	float radius;
	array<int, NCHILD> children;
	int src_begin;
	int src_end;
	int snk_begin;
	int snk_end;
	multipole multi;
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
		return nodes[i].src_begin;
	}
	CUDA_EXPORT inline int get_pend(int i) const {
		return nodes[i].src_end;
	}
	CUDA_EXPORT inline int get_snk_begin(int i) const {
		return nodes[i].snk_begin;
	}
	CUDA_EXPORT inline int get_snk_end(int i) const {
		return nodes[i].snk_end;
	}
	CUDA_EXPORT inline multipole get_multipole(int i) const {
		return nodes[i].multi;
	}
	CUDA_EXPORT inline float get_mass(int i) const {
		return nodes[i].multi[0];
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
	inline void adjust_src_indexes(int dif) {
		for( int i = 0; i < sz; i++) {
			nodes[i].src_begin += dif;
			nodes[i].src_end += dif;
		}
	}
	inline void adjust_snk_indexes(int dif) {
		for( int i = 0; i < sz; i++) {
			nodes[i].snk_begin += dif;
			nodes[i].snk_end += dif;
		}
	}
}
;

std::pair<tree, vector<sink_bucket>> tree_create(const array<int, NDIM>& cell_index, chaincell cell);

#endif /* TREE_HPP_ */
