#include <tigerpm/tree.hpp>
#include <tigerpm/particles.hpp>
#include <tigerpm/util.hpp>

#define MAX_DEPTH 50

static int sort(tree& t, vector<sink_bucket>& sink_buckets, const range<double>& box, int begin, int end, int depth, bool sunk = false) {
	tree_node node;
	for (int dim = 0; dim < NDIM; dim++) {
		if (box.end[dim] < 0.0) {
			PRINT("%e %e\n", box.begin[dim], box.end[dim]);
			assert(false);
		}
	}

	if (depth > MAX_DEPTH) {
		PRINT("Tree depth exceeded - two identical particles ? \n");
		abort();
	}
	int index = t.allocate();
//	PRINT( "%i %i\n", begin, end);
	node.src_begin = begin;
	node.src_end = end;
	if (end - begin <= std::min(SOURCE_BUCKET_SIZE, SINK_BUCKET_SIZE)) {
//		PRINT( "END %i %i\n", begin, end);
		float mass = end - begin;
		node.children[0] = -1;
		node.children[1] = -1;
		array<double, NDIM> x;
		array<double, NDIM> xmin;
		array<double, NDIM> xmax;
		for (int dim = 0; dim < NDIM; dim++) {
			x[dim] = 0.0;
			xmin[dim] = 1.0;
			xmax[dim] = 0.0;
		}
		for (int i = begin; i < end; i++) {
			for (int dim = 0; dim < NDIM; dim++) {
				const auto x = particles_pos(dim, i).to_double();
//				if( x > box.end[dim] ) {
//					PRINT( "%e %e\n", x, box.end[dim]);
//				}
				if (x > box.end[dim]) {
					PRINT("%e %e\n", x, box.end[dim]);
				}
				assert(x <= box.end[dim]);
				assert(x >= box.begin[dim]);
				xmax[dim] = std::max(xmax[dim], x);
				xmin[dim] = std::min(xmin[dim], x);
			}
		}
		if (mass > 0.0) {
			for (int dim = 0; dim < NDIM; dim++) {
				x[dim] = (xmax[dim] + xmin[dim]) * 0.5;
				assert(x[dim] <= box.end[dim]);
				assert(x[dim] >= box.begin[dim]);
			}
		} else {
			for (int dim = 0; dim < NDIM; dim++) {
				x[dim] = (box.begin[dim] + box.end[dim]) * 0.5;
				assert(x[dim] <= box.end[dim]);
				assert(x[dim] >= box.begin[dim]);
			}
		}
		multipole m;
		for (int i = 0; i < MULTIPOLE_SIZE; i++) {
			m[i] = 0.0f;
		}
		for (int i = begin; i < end; i++) {
			array<float, NDIM> dx;
			for (int dim = 0; dim < NDIM; dim++) {
				dx[dim] = particles_pos(dim, i).to_double() - x[dim];
			}
			const auto this_pole = P2M_kernel(dx);
			for (int j = 0; j < MULTIPOLE_SIZE; j++) {
				m[j] += this_pole[j];
			}
		}
		node.multi = m;
		double r2max = 0.0;
		for (int dim = 0; dim < NDIM; dim++) {
			node.x[dim] = x[dim];
		}
		for (int i = begin; i < end; i++) {
			double r2 = 0.0;
			for (int dim = 0; dim < NDIM; dim++) {
				node.x[dim] = x[dim];
				r2 += sqr(x[dim] - particles_pos(dim, i).to_double());
			}
			r2max = std::max(r2max, r2);
		}
		node.radius = std::sqrt(r2max) + get_options().hsoft;
	} else {
		const int long_dim = box.longest_dim();
		const auto child_boxes = box.split();
		for (int dim = 0; dim < NDIM; dim++) {
			if (child_boxes.second.end[dim] < 0.0) {
				PRINT("%e %e\n", box.begin[dim], box.end[dim]);
				PRINT("%e %e\n", child_boxes.first.begin[dim], child_boxes.first.end[dim]);
				assert(false);
			}
		}
		const int mid = particles_sort(begin, end, child_boxes.first.end[long_dim], long_dim);
		//	PRINT( "%i %i %i\n", begin, mid, end);
		bool this_sunk = end - begin <= SINK_BUCKET_SIZE;
		node.children[0] = sort(t, sink_buckets, child_boxes.first, begin, mid, depth + 1, this_sunk);
		node.children[1] = sort(t, sink_buckets, child_boxes.second, mid, end, depth + 1, this_sunk);
		const int i0 = node.children[0];
		const int i1 = node.children[1];
		float mass = t.get_mass(i0) + t.get_mass(i1);
		array<double, NDIM> n;
		array<double, NDIM> x;
		for (int dim = 0; dim < NDIM; dim++) {
			n[dim] = t.get_x(dim, i1).to_double() - t.get_x(dim, i0).to_double();
		}
		const auto norminv = 1.0 / std::sqrt(sqr(n[0], n[1], n[2]));
		for (int dim = 0; dim < NDIM; dim++) {
			n[dim] *= norminv;
		}
		for (int dim = 0; dim < NDIM; dim++) {
			if (t.get_mass(i0) == 0.0) {
				x[dim] = t.get_x(dim, i0).to_double();
			} else if (t.get_mass(i1) == 0.0) {
				x[dim] = t.get_x(dim, i1).to_double();
			} else {
				x[dim] = (t.get_x(dim, i0).to_double() + t.get_x(dim, i1).to_double() + n[dim] * (t.get_radius(i1) - t.get_radius(i0))) * 0.5;
			}
			if (x[dim] > box.end[dim]) {
				PRINT("! %e %e %e %e %e\n", box.begin[dim], x[dim], box.end[dim], t.get_x(dim, i0).to_double(), t.get_x(dim, i1).to_double());
			}
			if (x[dim] <= box.begin[dim]) {
				PRINT("! %e %e %e %e %e\n", box.begin[dim], x[dim], box.end[dim], t.get_x(dim, i0).to_double(), t.get_x(dim, i1).to_double());
			}
//			assert(x[dim] <= box.end[dim]);
//			assert(x[dim] >= box.begin[dim]);
		}
		multipole m, m1;
		array<float, NDIM> dx0, dx1;
		for (int dim = 0; dim < NDIM; dim++) {
			dx0[dim] = t.get_x(dim, i0).to_double() - x[dim];
			dx1[dim] = t.get_x(dim, i1).to_double() - x[dim];
		}
		m = M2M_kernel(t.get_multipole(i0), dx0);
		m1 = M2M_kernel(t.get_multipole(i1), dx1);
		for (int i = 0; i < MULTIPOLE_SIZE; i++) {
			m[i] += m1[i];
		}
		node.multi = m;
		for (int dim = 0; dim < NDIM; dim++) {
			node.x[dim] = x[dim];
		}
		double r20 = 0.0;
		double r21 = 0.0;
		for (int dim = 0; dim < NDIM; dim++) {
			if (t.get_mass(i0)) {
				r20 += sqr(t.get_x(dim, i0).to_double() - x[dim]);
			}
			if (t.get_mass(i1)) {
				r21 += sqr(t.get_x(dim, i1).to_double() - x[dim]);
			}
		}
		const auto r2_bbb = sqr(box.begin[0] - x[0], box.begin[1] - x[1], box.begin[2] - x[2]);
		const auto r2_bbe = sqr(box.begin[0] - x[0], box.begin[1] - x[1], box.end[2] - x[2]);
		const auto r2_beb = sqr(box.begin[0] - x[0], box.end[1] - x[1], box.begin[2] - x[2]);
		const auto r2_bee = sqr(box.begin[0] - x[0], box.end[1] - x[1], box.end[2] - x[2]);
		const auto r2_ebb = sqr(box.end[0] - x[0], box.begin[1] - x[1], box.begin[2] - x[2]);
		const auto r2_ebe = sqr(box.end[0] - x[0], box.begin[1] - x[1], box.end[2] - x[2]);
		const auto r2_eeb = sqr(box.end[0] - x[0], box.end[1] - x[1], box.begin[2] - x[2]);
		const auto r2_eee = sqr(box.end[0] - x[0], box.end[1] - x[1], box.end[2] - x[2]);
		const auto r2_max = std::max(std::max(std::max(r2_bbb, r2_bbe), std::max(r2_beb, r2_bee)), std::max(std::max(r2_ebb, r2_ebe), std::max(r2_eeb, r2_eee)));
		node.radius = std::min(std::max(std::sqrt(r20) + t.get_radius(i0), std::sqrt(r21) + t.get_radius(i1)), std::sqrt(r2_max) + get_options().hsoft);
	}
	if (end - begin <= SINK_BUCKET_SIZE && !sunk) {
		sink_bucket bucket;
		array<double, NDIM> x;
		array<double, NDIM> xmin;
		array<double, NDIM> xmax;
		for (int dim = 0; dim < NDIM; dim++) {
			x[dim] = 0.0;
			xmin[dim] = 1.0;
			xmax[dim] = 0.0;
		}
		for (int i = begin; i < end; i++) {
			for (int dim = 0; dim < NDIM; dim++) {
				const auto x = particles_pos(dim, i).to_double();
				xmax[dim] = std::max(xmax[dim], x);
				xmin[dim] = std::min(xmin[dim], x);
			}
		}
		if (end - begin) {
			for (int dim = 0; dim < NDIM; dim++) {
				x[dim] = (xmax[dim] + xmin[dim]) * 0.5;
			}
		} else {
			for (int dim = 0; dim < NDIM; dim++) {
				x[dim] = (box.begin[dim] + box.end[dim]) * 0.5;
			}
		}
		for (int i = begin; i < end; i++) {
			array<float, NDIM> dx;
			for (int dim = 0; dim < NDIM; dim++) {
				dx[dim] = particles_pos(dim, i).to_double() - x[dim];
			}
		}
		double r2max = 0.0;
		for (int i = begin; i < end; i++) {
			double r2 = 0.0;
			for (int dim = 0; dim < NDIM; dim++) {
				bucket.x[dim] = x[dim];
				r2 += sqr(x[dim] - particles_pos(dim, i).to_double());
			}
			r2max = std::max(r2max, r2);
		}
		bucket.radius = std::sqrt(r2max) + get_options().hsoft;
		bucket.snk_begin = bucket.src_begin = node.src_begin;
		bucket.snk_end = bucket.src_end = node.src_end;
		sink_buckets.push_back(bucket);
	}
	node.snk_begin = node.src_begin;
	node.snk_end = node.src_end;
//	PRINT("%e\n", node.radius);
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
		int i = cell_index[dim];
		if (i >= Nchain) {
			i -= Nchain;
		} else if (i < 0) {
			i += Nchain;
		}
		box.begin[dim] = i * Ninv;
		box.end[dim] = (i + 1) * Ninv;
		if (box.end[dim] > 1.0 || box.begin[dim] > 1.0) {
			box.end[dim] -= 1.0;
			box.begin[dim] -= 1.0;
		} else if (box.end[dim] < 0.0 || box.begin[dim] < 0.0) {
			box.end[dim] += 1.0;
			box.begin[dim] += 1.0;
		}
		if (box.end[dim] < 0.0) {
			PRINT("%i %i %e %e -----\n", i, cell_index[dim], box.begin[dim], box.end[dim]);
		}
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
			assert(false);
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

tree tree::to_device() const {
	tree t;
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

