#include <tigerpm/chainmesh.hpp>
#include <tigerpm/particles.hpp>
#include <tigerpm/util.hpp>
#include <tigerpm/hpx.hpp>

struct indices_hash {
	inline size_t operator()(const array<int, NDIM>& indices) const {
		std::hash<int> ihash;
		size_t hash = 0;
		for (int dim = 0; dim < NDIM; dim++) {
			hash ^= ihash(indices[dim]) << (dim * 10);
		}
		return hash;
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
using part_map_type = std::unordered_map<array<int,NDIM>, std::vector<particle>, indices_hash, indices_equal>;

static cell_map_type cells;
static part_map_type bnd_parts;
static mutex_type mutex;
static range<int> mybox;

static void sort(const range<int> chain_box, int pbegin, int pend);
static void transmit_chain_particles(array<int, NDIM>, vector<particle>);

HPX_PLAIN_ACTION(chainmesh_create);
HPX_PLAIN_ACTION(chainmesh_exchange_begin);
HPX_PLAIN_ACTION(chainmesh_exchange_end);
HPX_PLAIN_ACTION(transmit_chain_particles);

#define NPARTS_PER_LOCK 1024

chaincell chainmesh_get(const array<int, NDIM>& i) {
	assert(cells.find(i) != cells.end());
	return cells[i];
}

static void transmit_chain_particles(array<int, NDIM> celli, vector<particle> parts) {
	std::lock_guard<mutex_type> lock(mutex);
	bnd_parts[celli] = std::move(parts);
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
	spinlock_type this_mutex;
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
								for (i[2] = inter.begin[2]; i[2] != inter.end[2]; i[2]++) {
									these_futs.push_back(hpx::async([i,si,&futs,rank,&this_mutex]() {
										assert(cells.find(i) != cells.end());
										vector<particle> parts;
										const auto& cell = cells[i];
										parts.reserve(cell.pend - cell.pbegin);
										for (int k = cell.pbegin; k != cell.pend; k++) {
											assert(k >= 0);
											assert(k < particles_local_size());
											parts.push_back(particles_get_particle(k));
										}
										auto j = i;
										for (int dim = 0; dim < NDIM; dim++) {
											j[dim] -= si[dim];
										}
										auto fut = hpx::async<transmit_chain_particles_action>(hpx_localities()[rank], j,
												std::move(parts));
										std::lock_guard<spinlock_type> lock(this_mutex);
										futs.push_back(std::move(fut));
									}));
								}
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
		particles_resize(count);
	}
	for (auto i = bnd_parts.begin(); i != bnd_parts.end(); i++) {
		auto celli = i->first;
		auto func = [celli](std::vector<particle> parts) {
			const auto cell = cells[celli];
			for( int i = cell.pbegin; i != cell.pend; i++) {
				particles_set_particle(parts[i - cell.pbegin], i);
			}
		};
		futs.push_back(hpx::async(func, std::move(i->second)));
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
			assert(x >= chain_box.begin[dim] * Ninv);
			assert(x <= chain_box.end[dim] * Ninv);
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
