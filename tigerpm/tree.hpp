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
	int pbegin;
	int pend;
};

class tree {
	tree_node* nodes;
	int size;
	int cap;
	bool device;
	void resize(int new_size) {
		if (new_size > cap) {
			cap = 1;
			while (cap < new_size) {
				cap *= 2;
			}
			tree_node* new_nodes = new tree_node[cap];
			if (nodes) {
				std::memcpy(new_nodes, nodes, size * sizeof(tree_node));
				delete[] nodes;
			}
			nodes = new_nodes;
		}
		size = new_size;
	}
public:
	tree() {
		nodes = nullptr;
		size = cap = 0;
		device = false;
	}
	~tree() {
		if (nodes) {
			if (!device) {
				delete[] nodes;
			} else {
				CUDA_CHECK(cudaFree(nodes));
			}
		}
	}
	tree(const tree&) = delete;
	tree& operator=(const tree&) = delete;
	tree(tree&&) = default;
	tree& operator=(tree&&) = default;
	tree to_device(cudaStream_t stream) const {
		tree t;
		CUDA_CHECK(cudaMallocAsync(&t.nodes, sizeof(tree_node) * size, stream));
		CUDA_CHECK(cudaMemcpyAsync(t.nodes, nodes, sizeof(tree_node) * size, cudaMemcpyHostToDevice, stream));
		t.size = t.cap = size;
		t.device = true;
		return t;
	}
	int allocate() {
		int index = size;
		size++;
		return index;
	}
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
	void set(tree_node node, int i) {
		nodes[i] = node;
	}
};

std::pair<tree, vector<sink_bucket>> tree_create(const array<int, NDIM>& cell_index, chaincell cell);

#endif /* TREE_HPP_ */
