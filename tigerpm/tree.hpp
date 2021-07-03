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

struct tree_node {
	array<fixed32, NDIM> x;
	float mass;
	float radius;
	array<int, NCHILD> children;
	int pbegin;
	int pend;
};

struct sink_bucket {
	int src_begin;
	int src_end;
	int snk_begin;
	int snk_end;
	float radius;
};

class tree {
	tree_node* nodes;
	int sz;
	int cap;
	bool device;
	void resize(int new_size);
public:
	size_t size() const {
		return sz * sizeof(tree_node) + sizeof(tree);
	}
	tree();
	~tree();
	tree& operator=(const tree& other);
	tree & operator=(tree && other);
	tree(const tree& other);
	tree(tree && other);
	tree to_device(cudaStream_t stream) const;
	int allocate();
	CUDA_EXPORT inline
	fixed32 get_x(int dim, int i) const {
		return nodes[i].x[dim];
	}
	CUDA_EXPORT inline
	float get_mass(int i) const {
		return nodes[i].mass;
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
