#define PARTICLES_CPP

#include <tigerpm/fixed.hpp>

std::array<std::vector<fixed32>, NDIM> particles_X;
std::array<std::vector<float>, NDIM> particles_U;
std::vector<char> particles_R;
#ifdef FORCE_TEST
std::vector<float> particles_P;
std::array<std::vector<float>, NDIM> particles_G;
#endif

using map_type = std::unordered_map<int, int>;

#include <hpx/serialization/unordered_map.hpp>

#include <tigerpm/options.hpp>
#include <tigerpm/particles.hpp>
#include <tigerpm/hpx.hpp>
#include <tigerpm/range.hpp>
#include <tigerpm/util.hpp>

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
static auto& X = particles_X;
static auto& U = particles_U;
static auto& R = particles_R;
#ifdef FORCE_TEST
static auto& P = particles_P;
static auto& G = particles_G;
#endif

static std::vector<int> free_indices;
static std::vector<std::vector<particle>> recv_parts;
static spinlock_type send_mutex;
static spinlock_type recv_mutex;
static int sort_counter = 0;

static void domain_sort_end();
static void domain_sort_begin();
static int find_domain(std::array<int, NDIM> I);
static void find_domains(domain_t*);
static void transmit_particles(std::vector<particle>);
static std::unordered_map<int, int> get_particles_per_rank();
static std::vector<particle> get_particles_sample(std::vector<int> sample_counts);

HPX_PLAIN_ACTION (domain_sort_end);
HPX_PLAIN_ACTION (domain_sort_begin);
HPX_PLAIN_ACTION (transmit_particles);
HPX_PLAIN_ACTION (particles_random_init);
HPX_PLAIN_ACTION (get_particles_per_rank);
HPX_PLAIN_ACTION (get_particles_sample);

void particles_domain_sort() {
	domain_sort_begin();
	domain_sort_end();
}

range<int> particles_get_local_box() {
	return find_my_box(get_options().chain_dim);
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

std::array<int, NDIM> particles_mesh_loc(int index) {
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
	const auto mybox = find_my_box(get_options().chain_dim);
	std::unordered_map<int, std::vector<particle>> sends;
	PRINT("Domain sort begin on %i\n", hpx_rank());
	const int nthreads = hpx::thread::hardware_concurrency();
	for (int proc = 0; proc < nthreads; proc++) {
		futs2.push_back(hpx::async([proc,nthreads,mybox,&sends,&futs1]() {
			const int begin = size_t(proc) * size_t(particles_size()) / size_t(nthreads);
			const int end = size_t(proc + 1) * size_t(particles_size()) / size_t(nthreads);
			for (int i = begin; i < end; i++) {
				const auto mi = particles_mesh_loc(i);
				if (!mybox.contains(mi)) {
					const auto rank = find_domain(mi);
					assert(rank != hpx_rank());
					std::unique_lock<spinlock_type> lock(send_mutex);
					auto& entry = sends[rank];
					entry.push_back(particles_get_particle(i));
					free_indices.push_back(i);
					if (entry.size() == MAX_PARTS_PER_MSG) {
						PRINT("Sending %i particles from %i to %i\n", entry.size(), hpx_rank(), rank);
						auto data = std::move(entry);
						lock.unlock();
						futs1.push_back(hpx::async < transmit_particles_action > (hpx_localities()[rank], std::move(data)));
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
	R.resize(new_size);
#ifdef FORCE_TEST
	for (int dim = 0; dim < NDIM; dim++) {
		G[dim].resize(new_size);
	}
	P.resize(new_size);
#endif
}

static std::unordered_map<int, int> get_particles_per_rank() {
	std::vector < hpx::future<std::unordered_map<int, int>>>futs;
	for (auto c : hpx_children()) {
		futs.push_back(hpx::async < get_particles_per_rank_action > (c));
	}

	std::unordered_map<int, int> rc;
	rc[hpx_rank()] = particles_size();

	for (auto& fut : futs) {
		auto data = fut.get();
		for (auto i = data.begin(); i != data.end(); i++) {
			rc[i->first] = i->second;
		}
	}

	return rc;
}

std::vector<int> particles_per_rank() {
	auto data = get_particles_per_rank_action()(hpx_localities()[0]);
	std::vector<int> rc;
	for (int i = 0; i < hpx_size(); i++) {
		rc[i] = data[i];
	}
	return rc;
}

static std::vector<particle> get_particles_sample(std::vector<int> sample_counts) {
	std::vector < hpx::future<std::vector<particle>>>futs;
	for (auto c : hpx_children()) {
		futs.push_back(hpx::async < get_particles_sample_action > (c, sample_counts));
	}

	std::set<int> indices;
	std::vector<particle> samples;
	for (int i = 0; i < sample_counts[hpx_rank()]; i++) {
		const int index = rand() % particles_size();
		if (indices.find(index) == indices.end()) {
			indices.insert(index);
			samples.push_back(particles_get_particle(index));
		}
	}

	for (int i = 0; i < futs.size(); i++) {
		const auto other = futs[i].get();
		samples.insert(samples.end(), other.begin(), other.end());
	}

	return samples;

}

std::vector<particle> particles_sample(const std::vector<int>& sample_counts) {
	return get_particles_sample_action()(hpx_localities()[0], sample_counts);
}

std::vector<particle> particles_sample(int Nsamples) {
	auto parts_per_rank = particles_per_rank();
	size_t total_parts = 0;
	for (int i = 0; i < hpx_size(); i++) {
		total_parts += parts_per_rank[i];
	}
	std::vector<int> samples_per_proc(hpx_size(), 0);
	for (int i = 0; i < Nsamples; i++) {
		const size_t index = (size_t(rand()) * size_t(rand())) % total_parts;
		size_t total = 0;
		int proc_index = -1;
		while (index >= total) {
			proc_index++;
			total += parts_per_rank[proc_index];
		}
		samples_per_proc[proc_index]++;
	}
	return particles_sample(samples_per_proc);
}

