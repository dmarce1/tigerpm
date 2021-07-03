#include <tigerpm/tree.hpp>
#include <tigerpm/particles.hpp>
#include <tigerpm/util.hpp>

#define MAX_DEPTH 50

static int sort(tree& t, vector<sink_bucket>& sink_buckets, const range<double>& box, int begin, int end, int depth) {
	tree_node node;
	if (depth > MAX_DEPTH) {
		PRINT("Tree depth exceeded - two identical particles ? \n");
		abort();
	}
	int index = t.allocate();
//	PRINT( "%i %i\n", begin, end);
	if (end - begin <= BUCKET_SIZE) {
//		PRINT( "END %i %i\n", begin, end);
		node.mass = begin - end;
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
		if (node.mass > 0.0) {
			for (int dim = 0; dim < NDIM; dim++) {
				x[dim] /= node.mass;
			}
		}
		double r2max = 0.0;
		for (int i = begin; i < end; i++) {
			double r2 = 0.0;
			for (int dim = 0; dim < NDIM; dim++) {
				node.x[dim] = x[dim];
				r2 += sqr(x[dim] - particles_pos(dim, i).to_double());
			}
			r2max = std::max(r2max, r2);
		}
		node.radius = std::sqrt(r2max) + get_options().hsoft;
		sink_bucket bucket;
		bucket.snk_begin = bucket.src_begin = begin;
		bucket.snk_end = bucket.src_end = end;
		bucket.radius = node.radius;
		bucket.x = node.x;
		sink_buckets.push_back(bucket);
	} else {
		const int long_dim = box.longest_dim();
		const auto child_boxes = box.split();
		const int mid = particles_sort(begin, end, child_boxes.first.end[long_dim], long_dim);
		//	PRINT( "%i %i %i\n", begin, mid, end);
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
	t.set(node, index);
	return index;
}

std::pair<tree, vector<sink_bucket>> tree_create(const array<int, NDIM>& cell_index, chaincell cell) {
	tree new_tree;
	vector<sink_bucket> sink_buckets;
	range<double> box;
	const double Nchain = get_options().chain_dim;
	const double Ninv = 1.0 / Nchain;
	for (int dim = 0; dim < NDIM; dim++) {
		box.begin[dim] = cell_index[dim] * Ninv;
		box.end[dim] = (cell_index[dim] + 1) * Ninv;
	}
	sort(new_tree, sink_buckets, box, cell.pbegin, cell.pend, 0);
	return std::make_pair(std::move(new_tree), std::move(sink_buckets));
}

tree::tree() {
	nodes = nullptr;
	sz = cap = 0;
	device = false;
}

tree::~tree() {
	if (nodes) {
		if (!device) {
			delete[] nodes;
		} else {
			CUDA_CHECK(cudaFree(nodes));
		}
	}
}

tree& tree::operator=(const tree& other) {
	device = other.device;
	if (device) {
		if (other.sz) {
			CUDA_CHECK(cudaMalloc(&nodes, sizeof(tree_node) * other.sz));
			CUDA_CHECK(cudaMemcpy(nodes, other.nodes, sizeof(tree_node) * other.sz, cudaMemcpyDeviceToDevice));
			sz = cap = other.sz;
		}
	} else {
		if (other.sz) {
			nodes = new tree_node[other.sz];
			std::memcpy(nodes, other.nodes, sizeof(tree_node) * other.sz);
			sz = cap = other.sz;
		}
	}
	return *this;
}

tree & tree::operator=(tree && other) {
	device = other.device;
	sz = other.sz;
	cap = other.cap;
	if (nodes) {
		if (!device) {
			delete[] nodes;
		} else {
			CUDA_CHECK(cudaFree(nodes));
		}
	}
	nodes = other.nodes;
	other.nodes = nullptr;
	return *this;
}

tree::tree(const tree& other) {
	nodes = nullptr;
	sz = cap = 0;
	*this = other;
}

tree::tree(tree && other) {
	nodes = nullptr;
	sz = cap = 0;
	*this = std::move(other);
}

tree tree::to_device(cudaStream_t stream) const {
	tree t;
	CUDA_CHECK(cudaMalloc(&t.nodes, sizeof(tree_node) * sz));
	CUDA_CHECK(cudaMemcpyAsync(t.nodes, nodes, sizeof(tree_node) * sz, cudaMemcpyHostToDevice, stream));
	t.sz = t.cap = sz;
	t.device = true;
	return t;
}

int tree::allocate() {
	int index = sz;
	resize(sz + 1);
	return index;
}

void tree::resize(int new_size) {
	if (new_size > cap) {
		cap = 1;
		while (cap < new_size) {
			cap *= 2;
		}
		tree_node* new_nodes = new tree_node[cap];
		if (nodes) {
			std::memcpy(new_nodes, nodes, sz * sizeof(tree_node));
			delete[] nodes;
		}
		nodes = new_nodes;
	}
	sz = new_size;
}
