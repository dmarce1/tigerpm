#include <tigerpm/tree.hpp>
#include <tigerpm/particles.hpp>
#include <tigerpm/util.hpp>

#define MAX_DEPTH 40

static int sort(tree& t, vector<sink_bucket>& sink_buckets, const range<double>& box, int begin, int end, int depth) {
	tree_node node;
	if (depth > MAX_DEPTH) {
		PRINT("Tree depth exceeded - two identical particles ? \n");
		abort();
	}
	if (end - begin < BUCKET_SIZE) {
		node.radius = 0.0;
		node.mass = end - begin;
		node.children[0] = -1;
		node.children[1] = -1;
		array<double, NDIM> x;
		for (int dim = 0; dim < NDIM; dim++) {
			x[dim] = 0.0;
		}
		for (int i = begin; i < end; i++) {
			for (int dim = 0; dim < NDIM; dim++) {
				x[dim] += particles_pos(dim, i).to_double();
			}
		}
		for (int dim = 0; dim < NDIM; dim++) {
			x[dim] /= node.mass;
		}
		double r2max = 0.0;
		for (int i = begin; i < end; i++) {
			double r2 = 0.0;
			for (int dim = 0; dim < NDIM; dim++) {
				node.x[dim] = x[dim];
				r2 += sqr(particles_pos(dim, i).to_double() - x[dim]);
			}
			r2max = std::max(r2, r2max);
		}
		node.radius = std::sqrt(r2max) + get_options().hsoft;
		sink_bucket bucket;
		bucket.pbegin = begin;
		bucket.pend = end;
		sink_buckets.push_back(bucket);
	} else {
		const int long_dim = box.longest_dim();
		const auto child_boxes = box.split();
		const int mid = particles_sort(begin, end, child_boxes.first.end[long_dim], long_dim);
		node.children[0] = sort(t, sink_buckets, child_boxes.first, begin, mid, depth + 1);
		node.children[1] = sort(t, sink_buckets, child_boxes.second, mid, end, depth + 1);
		const int i0 = node.children[0];
		const int i1 = node.children[1];
		node.mass = t.get_mass(i0) + t.get_mass(i1);
		array<double, NDIM> x;
		for (int dim = 0; dim < NDIM; dim++) {
			x[dim] = t.get_mass(i0) * t.get_x(dim, i0).to_double() + t.get_mass(i1) * t.get_x(dim, i1).to_double();
		}
		for (int dim = 0; dim < NDIM; dim++) {
			x[dim] /= node.mass;
		}
		for (int dim = 0; dim < NDIM; dim++) {
			node.x[dim] = x[dim];
		}
		double r20 = 0.0;
		double r21 = 0.0;
		for (int dim = 0; dim < NDIM; dim++) {
			r20 += sqr(t.get_x(dim, i0).to_double() - x[dim]);
			r21 += sqr(t.get_x(dim, i1).to_double() - x[dim]);
		}
		node.radius = std::sqrt(std::max(r20, r21));
	}
	int index = t.allocate();
	t.set(node, index);
	return index;
}

std::pair<tree, vector<sink_bucket>> tree_create(const array<int, NDIM>& cell_index, chaincell cell, int depth) {
	tree new_tree;
	vector<sink_bucket> sink_buckets;
	range<double> box;
	const double Nchain = get_options().chain_dim;
	for (int dim = 0; dim < NDIM; dim++) {
		box.begin[dim] = cell_index[dim] * Nchain;
		box.end[dim] = (cell_index[dim] + 1) * Nchain;
	}
	sort(new_tree, sink_buckets, box, cell.pbegin, cell.pend, 0);
	return std::make_pair(std::move(new_tree), std::move(sink_buckets));
}

