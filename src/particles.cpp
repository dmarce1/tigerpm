#define PARTICLES_CPP

#include <tigerpm/fixed.hpp>

std::array<std::vector<fixed32>, NDIM> particles_X;
std::array<std::vector<float>, NDIM> particles_U;

#include <tigerpm/options.hpp>
#include <tigerpm/particles.hpp>
#include <tigerpm/hpx.hpp>
#include <tigerpm/range.hpp>

#include <unordered_map>
#include <gsl/gsl_rng.h>

#define MAX_PARTS_PER_MSG (4*1024*1024)

struct domain_t {
	domain_t* left;
	domain_t* right;
	int xdim;
	int mid;
	int proc;
};

static domain_t* root_domain = nullptr;
static std::array<std::vector<fixed32>, NDIM>& X = particles_X;
static std::array<std::vector<float>, NDIM>& U = particles_U;
static std::vector<int> free_indices;
static std::vector<std::vector<particle>> recv_parts;
static spinlock_type send_mutex;
static spinlock_type recv_mutex;
static int sort_counter = 0;

static void domain_sort_end();
static void domain_sort_begin();
static std::array<int, NDIM> mesh_loc(int index);
static int find_domain(std::array<int, NDIM> I);
static void find_domains(domain_t*);
static range<int> find_my_box();
static void transmit_particles(std::vector<particle>);

HPX_PLAIN_ACTION (domain_sort_end);
HPX_PLAIN_ACTION (domain_sort_begin);
HPX_PLAIN_ACTION (transmit_particles);
HPX_PLAIN_ACTION (particles_random_init);

void particles_domain_sort() {
	domain_sort_begin();
	domain_sort_end();
}

void particles_random_init() {
	std::vector<hpx::future<void>> futs;
	for (auto c : hpx_children()) {
		futs.push_back(hpx::async < particles_random_init_action > (c));
	}
	const size_t nparts = std::pow(size_t(get_options().part_dim), size_t(NDIM));
	const size_t begin = size_t(hpx_rank()) * nparts / size_t(hpx_size());
	const size_t end = size_t(hpx_rank() + 1) * nparts / size_t(hpx_size());
	const int size = end - begin;
	assert(size < std::numeric_limits<int>::max());
	particles_resize(size);
	const int nthreads = hpx::threads::hardware_concurrency();
	for (int proc = 0; proc < nthreads; proc++) {
		futs.push_back(hpx::async([proc,nthreads]() {
			const int seed = 4321*(hpx_size() * proc + hpx_rank() + 42);
			gsl_rng * rndgen = gsl_rng_alloc (gsl_rng_taus);
			assert(rndgen);
			gsl_rng_set(rndgen, seed);
			const int begin = size_t(proc) * particles_size() / size_t(nthreads);
			const int end = size_t(proc+1) * particles_size() / size_t(nthreads);
			for( int i = begin; i < end; i++) {
				for( int dim = 0; dim < NDIM; dim++) {
					particles_pos(dim,i) = gsl_rng_uniform(rndgen);
					particles_vel(dim,i) = 0.0f;
				}
			}
			gsl_rng_free(rndgen);
		}));
	}
	hpx::wait_all(futs.begin(), futs.end());

}

static std::array<int, NDIM> mesh_loc(int index) {
	static const double N = get_options().chain_dim;
	std::array<int, NDIM> i;
	for (int dim = 0; dim < NDIM; dim++) {
		i[dim] = X[dim][index].to_double() * N;
	}
	return i;
}

static void domain_sort_begin() {
	std::vector<hpx::future<void>> futs1;
	std::vector<hpx::future<void>> futs2;
	for (auto c : hpx_children()) {
		futs1.push_back(hpx::async < domain_sort_begin_action > (c));
	}
	if (root_domain == nullptr) {
		root_domain = new domain_t;
		find_domains(root_domain);
	}
	const auto mybox = find_my_box();
	std::unordered_map<int, std::vector<particle>> sends;
	PRINT("Domain sort begin on %i\n", hpx_rank());
	const int nthreads = hpx::thread::hardware_concurrency();
	for (int proc = 0; proc < nthreads; proc++) {
		futs2.push_back(hpx::async([proc,nthreads,mybox,&sends,&futs1]() {
			const int begin = size_t(proc) * size_t(particles_size()) / size_t(nthreads);
			const int end = size_t(proc + 1) * size_t(particles_size()) / size_t(nthreads);
			for (int i = begin; i < end; i++) {
				const auto mi = mesh_loc(i);
				if (!mybox.contains(mi)) {
					const auto rank = find_domain(mi);
					assert(rank != hpx_rank());
					std::lock_guard<spinlock_type> lock(send_mutex);
					auto& entry = sends[rank];
					entry.push_back(particles_get_particle(i));
					free_indices.push_back(i);
					//			printf( "%i %i %i - %i to %i\n", mi[0], mi[1], mi[2], hpx_rank(), rank);
				if (entry.size() == MAX_PARTS_PER_MSG) {
					PRINT("Sending %i particles from %i to %i\n", entry.size(), hpx_rank(), rank);
					futs1.push_back(hpx::async < transmit_particles_action > (hpx_localities()[rank], std::move(entry)));
				}
			}
		}
	}));
	}

	hpx::wait_all(futs2.begin(), futs2.end());
	for (auto i = sends.begin(); i != sends.end(); i++) {
		if (i->second.size()) {
			auto& entry = i->second;
			PRINT("Sending %i particles from %i to %i\n", entry.size(), hpx_rank(), i->first);
			futs1.push_back(hpx::async < transmit_particles_action > (hpx_localities()[i->first], std::move(entry)));

		}
	}

	hpx::wait_all(futs1.begin(), futs1.end());

}

static void domain_sort_end() {
	std::vector<hpx::future<void>> futs;
	for (auto c : hpx_children()) {
		futs.push_back(hpx::async < domain_sort_end_action > (c));
	}
	int total_recv = 0;
	for (int i = 0; i < recv_parts.size(); i++) {
		const auto& parts = recv_parts[i];
		total_recv += parts.size();
	}
	const int balance = total_recv - free_indices.size();
	if (balance > 0) {
		for (int i = 0; i < balance; i++) {
			free_indices.push_back(particles_size() + i);
		}
		particles_resize(particles_size() + balance);
	} else if (balance < 0) {
		std::sort(free_indices.begin(), free_indices.end());
		while (free_indices.size() != total_recv) {
			particles_set_particle(particles_get_particle(particles_size() - 1), free_indices.back());
			particles_resize(particles_size() - 1);
			free_indices.pop_back();
		}
	}
	int free_index = 0;
	PRINT("Received %i vectors on %i\n", recv_parts.size(), hpx_rank());
	for (int i = 0; i < recv_parts.size(); i++) {
		const int sz = recv_parts[i].size();
		futs.push_back(hpx::async([i,free_index]() {
			auto& parts = recv_parts[i];
			PRINT( "Adding %i parts on %i\n", parts.size(), hpx_rank());
			for( int j = 0; j < parts.size(); j++) {
				particles_set_particle(parts[j],free_indices[j+free_index]);
			}
			parts = std::vector<particle>();
		}));
		free_index += sz;
	}

	hpx::wait_all(futs.begin(), futs.end());
	free_indices = decltype(free_indices)();
	recv_parts = decltype(recv_parts)();

}

static void transmit_particles(std::vector<particle> parts) {
	std::lock_guard<spinlock_type> lock(recv_mutex);
	recv_parts.push_back(std::move(parts));
}

static range<int> find_my_box(range<int> box, int begin, int end) {
	if (end - begin == 1) {
		return box;
	} else {
		const int xdim = box.longest_dim();
		auto left = box;
		auto right = box;
		const int mid = (begin + end) / 2;
		const double w = double(mid - begin) / double(end - begin);
		left.end[xdim] = right.begin[xdim] = (((1.0 - w) * box.begin[xdim] + w * box.end[xdim]) + 0.5);
		if (hpx_rank() < mid) {
			assert(hpx_rank() >= begin);
			return find_my_box(left, begin, mid);
		} else {
			assert(hpx_rank() >= mid);
			assert(hpx_rank() < end);
			return find_my_box(right, mid, end);
		}
	}
}

static int find_domain(std::array<int, NDIM> I) {
	domain_t* tree = root_domain;
	while (tree->left != nullptr) {
		if (I[tree->xdim] < tree->mid) {
			tree = tree->left;
		} else {
			tree = tree->right;
		}
	}
	assert(tree->right == nullptr);
	return tree->proc;
}

static void find_domains(domain_t* tree, range<int> box, int begin, int end) {
	if (end - begin == 1) {
		tree->left = tree->right = nullptr;
		tree->proc = begin;
	} else {
		const int xdim = box.longest_dim();
		auto left = box;
		auto right = box;
		const int mid = (begin + end) / 2;
		const double w = double(mid - begin) / double(end - begin);
		left.end[xdim] = right.begin[xdim] = (((1.0 - w) * box.begin[xdim] + w * box.end[xdim]) + 0.5);
		tree->left = new domain_t;
		tree->right = new domain_t;
		tree->xdim = xdim;
		tree->mid = left.end[xdim];
		find_domains(tree->left, left, begin, mid);
		find_domains(tree->right, right, mid, end);
	}
}

static void find_domains(domain_t* tree) {
	range<int> box(get_options().chain_dim);
	return find_domains(tree, box, 0, hpx_size());
}

static range<int> find_my_box() {
	return find_my_box(range<int>(get_options().chain_dim), 0, hpx_size());
}

int particles_size() {
	return X[0].size();
}

void particles_resize(size_t new_size) {
	if (new_size > std::numeric_limits<int>::max()) {
		PRINT("Error - particle set size exceeds %li\n", std::numeric_limits<int>::max());
		abort();
	}
	for (int dim = 0; dim < NDIM; dim++) {
		X[dim].resize(new_size);
		U[dim].resize(new_size);
	}
}
