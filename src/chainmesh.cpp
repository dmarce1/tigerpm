#include <tigerpm/chainmesh.hpp>
#include <tigerpm/particles.hpp>
#include <tigerpm/util.hpp>
#include <tigerpm/hpx.hpp>

struct indices_hash {
	inline size_t operator()(const array<int, NDIM>& indices) const {
		std::hash<int> ihash;
		size_t hash1 = 1664525LL * indices[0] + 1013904223LL;
		size_t hash2 = 22695477LL * indices[1] + 1LL;
		size_t hash3 = 134775813LL * indices[1] + 1LL;
		return hash1 ^ hash2 ^ hash3;
	}
};

struct indices_equal {
	inline size_t operator()(const array<int, NDIM>& i1, const array<int, NDIM>& i2) const {
		bool rc = true;
		for (int dim = 0; dim < NDIM; dim++) {
			if (i1[dim] != i2[dim]) {
				rc = false;
				break;
			}
		}
		return rc;
	}
};

using cell_map_type = std::unordered_map<array<int,NDIM>, chaincell, indices_hash, indices_equal>;
using bnd_element_type = std::pair<array<int,NDIM>,vector<particle_pos>>;
static cell_map_type cells;
static vector<bnd_element_type> bnd_parts;
static mutex_type mutex;
static range<int> mybox;

static void sort(const range<int> chain_box, int pbegin, int pend);
static void transmit_chain_particles(vector<bnd_element_type>);

HPX_PLAIN_ACTION(chainmesh_create);
HPX_PLAIN_ACTION(chainmesh_exchange_begin);
HPX_PLAIN_ACTION(chainmesh_exchange_end);
HPX_PLAIN_ACTION(transmit_chain_particles);

#define NPARTS_PER_LOCK 1024

chaincell chainmesh_get(const array<int, NDIM>& i) {
	assert(cells.find(i) != cells.end());
	return cells[i];
}

static void transmit_chain_particles(vector<bnd_element_type> parts) {
	std::lock_guard<mutex_type> lock(mutex);
	for (int i = 0; i < parts.size(); i++) {
		bnd_parts.push_back(parts[i]);
	}
}

void chainmesh_exchange_begin() {
	const static int N = get_options().chain_dim;
	vector<hpx::future<void>> futs;
	for (auto c : hpx_children()) {
		futs.push_back(hpx::async<chainmesh_exchange_begin_action>(c));
	}
	vector<range<int>> allboxes(hpx_size());
	find_all_boxes(allboxes, N);
	const auto myvol = mybox.volume();
	mutex_type this_mutex;
	for (int rank = 0; rank < allboxes.size(); rank++) {
		const auto box = allboxes[rank];
		array<int, NDIM> si;
		for (si[0] = -N; si[0] <= +N; si[0] += N) {
			for (si[1] = -N; si[1] <= +N; si[1] += N) {
				for (si[2] = -N; si[2] <= +N; si[2] += N) {
					const auto inter = box.pad(CHAIN_BW).shift(si).intersection(mybox);
					const auto vol = inter.volume();
					if (vol != 0 && vol != myvol) {
						array<int, NDIM> i;
						std::vector<hpx::future<void>> these_futs;
						for (i[0] = inter.begin[0]; i[0] != inter.end[0]; i[0]++) {
							for (i[1] = inter.begin[1]; i[1] != inter.end[1]; i[1]++) {
								these_futs.push_back(hpx::async([inter,si,&futs,rank,&this_mutex](array<int,NDIM> i) {
									vector<bnd_element_type> many_parts;
									for (i[2] = inter.begin[2]; i[2] != inter.end[2]; i[2]++) {
										assert(cells.find(i) != cells.end());
										vector<particle_pos> parts;
										const auto& cell = cells[i];
										parts.reserve(cell.pend - cell.pbegin);
										for (int k = cell.pbegin; k != cell.pend; k++) {
											assert(k >= 0);
											assert(k < particles_local_size());
											parts.push_back(particles_get_particle_pos(k));
										}
										auto j = i;
										for (int dim = 0; dim < NDIM; dim++) {
											j[dim] -= si[dim];
										}
										many_parts.push_back(std::make_pair(j,std::move(parts)));
									}
									auto fut = hpx::async<transmit_chain_particles_action>(hpx_localities()[rank],
											std::move(many_parts));
									std::lock_guard<mutex_type> lock(this_mutex);
									futs.push_back(std::move(fut));
								}, i));
							}
						}
						hpx::wait_all(these_futs.begin(), these_futs.end());
					}
				}
			}
		}
	}
	hpx::wait_all(futs.begin(), futs.end());
}

void chainmesh_exchange_end() {
	vector<hpx::future<void>> futs;
	for (auto c : hpx_children()) {
		futs.push_back(hpx::async<chainmesh_exchange_end_action>(c));
	}
	int count = particles_size();
	for (auto i = bnd_parts.begin(); i != bnd_parts.end(); i++) {
		cells[i->first].pbegin = count;
		count += i->second.size();
		cells[i->first].pend = count;
		particles_resize_pos(count);
	}
	const int nthreads = hpx::thread::hardware_concurrency();
	for (int proc = 0; proc < nthreads; proc++) {
		auto func = [proc, nthreads]() {
			for (auto i = proc; i < bnd_parts.size(); i+=nthreads) {
				auto celli = bnd_parts[i].first;
				const auto cell = cells[celli];
				for( int j = cell.pbegin; j != cell.pend; j++) {
					particles_set_particle_pos(bnd_parts[i].second[j - cell.pbegin], j);
				}
			}
		};
		futs.push_back(hpx::async(func));
	}
	hpx::wait_all(futs.begin(), futs.end());
	bnd_parts = decltype(bnd_parts)();
}

void chainmesh_create() {
	vector<hpx::future<void>> futs;
	for (auto c : hpx_children()) {
		futs.push_back(hpx::async<chainmesh_create_action>(c));
	}
	static int N = get_options().chain_dim;
	mybox = find_my_box(N);
	cells = decltype(cells)();
	sort(mybox, 0, particles_size());
#ifndef NDEBUG
	for (auto i = cells.begin(); i != cells.end(); i++) {
		for (auto j = cells.begin(); j != cells.end(); j++) {
			if (i != j) {
				if (std::min(i->second.pend, j->second.pend) > std::max(i->second.pbegin, j->second.pbegin)) {
					PRINT("%i %i\n", i->second.pbegin, i->second.pend);
					PRINT("%i %i\n", j->second.pbegin, j->second.pend);
					assert(false);
				}
			}
		}
	}
#endif
	hpx::wait_all(futs.begin(), futs.end());
}

range<int> chainmesh_interior_box() {
	static int N = get_options().chain_dim;
	return find_my_box(N);
}

static void sort(const range<int> chain_box, int pbegin, int pend) {
	int minthreadparts = std::max(1, (int) (particles_size() / hpx::thread::hardware_concurrency() / 8));
	static int N = get_options().chain_dim;
	static double Ninv = 1.0 / N;
	const int vol = chain_box.volume();
#ifndef NDEBUG
	for (int i = pbegin; i < pend; i++) {
		for (int dim = 0; dim < NDIM; dim++) {
			const auto x = particles_pos(dim, i).to_double();
			assert(x >= 0.9999999*chain_box.begin[dim] * Ninv);
			assert(x <= 1.0000001*chain_box.end[dim] * Ninv);
		}
	}
#endif
	if (vol == 1) {
		std::unique_lock<mutex_type> lock(mutex);
		auto& cell = cells[chain_box.begin];
		cell.pbegin = pbegin;
		cell.pend = pend;
		assert(cells.find(chain_box.begin) != cells.end());
	} else {
		int long_dim;
		int long_span = -1;
		for (int dim = NDIM - 1; dim >= 0; dim--) {
			const int span = chain_box.end[dim] - chain_box.begin[dim];
			if (span > 1) {
				long_span = span;
				long_dim = dim;
			}
		}
		const int mid_box = chain_box.begin[long_dim] + long_span / 2;
		const double mid_x = mid_box * Ninv;
		auto chain_box_left = chain_box;
		auto chain_box_right = chain_box;
		chain_box_left.end[long_dim] = chain_box_right.begin[long_dim] = mid_box;
		const auto pmid = particles_sort(pbegin, pend, mid_x, long_dim);
		if (pend - pbegin > minthreadparts) {
			auto futl = hpx::async(sort, chain_box_left, pbegin, pmid);
			sort(chain_box_right, pmid, pend);
			futl.get();
		} else {
			sort(chain_box_left, pbegin, pmid);
			sort(chain_box_right, pmid, pend);
		}
	}
}
