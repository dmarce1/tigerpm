#include <tigerpm/chainmesh.hpp>
#include <tigerpm/particles.hpp>
#include <tigerpm/util.hpp>
#include <tigerpm/hpx.hpp>

static std::vector<chaincell> cells;
static range<int> mybox;
static range<int> mybigbox;
static int thread_vol;
static shared_mutex_type mutex;

static void sort(const range<int> chain_box, int pbegin, int pend);

void transmit_chain_particles(std::vector<particle> parts);

HPX_PLAIN_ACTION (transmit_chain_particles);
HPX_PLAIN_ACTION (chainmesh_exchange_boundaries);
HPX_PLAIN_ACTION (chainmesh_create);

#define NPARTS_PER_LOCK 1000

void transmit_chain_particles(std::vector<particle> parts) {
	PRINT( "%i receiving %i particles\n", hpx_rank(), parts.size());
	std::unique_lock<shared_mutex_type> lock(mutex);
	const int offset = particles_size();
	particles_resize(particles_size() + parts.size());
	lock.unlock();
	const int nthreads = hpx::thread::hardware_concurrency();
	std::vector<hpx::future<void>> futs;
	for (int proc = 0; proc < nthreads; proc++) {
		const auto func = [proc, nthreads, offset, &parts]() {
			const int begin = size_t(proc) * size_t(parts.size()) / size_t(nthreads);
			const int end = size_t(proc+1) * size_t(parts.size()) / size_t(nthreads);
			for (int i = begin; i < end; i += NPARTS_PER_LOCK) {
				const int maxi = std::min(end, i + NPARTS_PER_LOCK);
				mutex.lock_shared();
				for (int j = i; j < maxi; j++) {
					particles_set_particle(parts[j], j + offset);
				}
				mutex.unlock_shared();
			}};
		futs.push_back(hpx::async(func));
	}
	hpx::wait_all(futs.begin(), futs.end());
}

void chainmesh_exchange_boundaries() {
	std::vector<hpx::future<void>> futs;
	std::vector<hpx::future<void>> futs2;
	for (auto c : hpx_children()) {
		futs.push_back(hpx::async < chainmesh_exchange_boundaries_action > (c));
	}
	std::vector<range<int>> allboxes(hpx_size());
	std::vector<range<int>> interboxes;
	std::vector<range<double>> interxboxes;
	std::vector<int> ranks;
	static int N = get_options().chain_dim;
	static double Ninv = 1.0 / N;
	find_all_boxes(allboxes, N);
	mybox = allboxes[hpx_rank()];
	for (int rank = 0; rank < allboxes.size(); rank++) {
		if (rank != hpx_rank()) {
			const auto inter = allboxes[rank].pad(1).intersection(mybox);
			if (inter.volume()) {
				range<double> interxbox;
				for (int dim = 0; dim < NDIM; dim++) {
					interxbox.begin[dim] = inter.begin[dim] * Ninv;
					interxbox.end[dim] = inter.end[dim] * Ninv;
				}
				interxboxes.push_back(interxbox);
				interboxes.push_back(inter);
				ranks.push_back(rank);
			}
		}
	}
	range<int> interiorbox = mybox.pad(-1);
	range<double> interiorx;
	for (int dim = 0; dim < NDIM; dim++) {
		interiorx.begin[dim] = interiorbox.begin[dim] * Ninv;
		interiorx.end[dim] = interiorbox.end[dim] * Ninv;
	}
	std::vector < std::vector < particle >> parts(ranks.size());
	std::vector<spinlock_type> mutexes(ranks.size());
	const int nthreads = hpx::thread::hardware_concurrency();
	for (int proc = 0; proc < nthreads; proc++) {
		const auto func = [proc,nthreads,&mutexes,&interiorx,&ranks,&interxboxes,&parts]() {
			const int begin = size_t(proc) * size_t(particles_local_size()) / size_t(nthreads);
			const int end = size_t(proc + 1) * size_t(particles_local_size()) / size_t(nthreads);
			for (int i = begin; i < end; i++) {
				std::array<double, NDIM> x;
				for (int dim = 0; dim < NDIM; dim++) {
					x[dim] = particles_pos(dim, i).to_double();
				}
				if (!interiorx.contains(x)) {
					const auto part = particles_get_particle(i);
					for (int rank = 0; rank < ranks.size(); rank++) {
						if (interxboxes[rank].contains(x)) {
							std::lock_guard<spinlock_type> lock(mutexes[rank]);
							parts[rank].push_back(part);
						}
					}
				}
			}
		};
		futs2.push_back(hpx::async(func));
	}
	hpx::wait_all(futs2.begin(), futs2.end());
	for (int rank = 0; rank < ranks.size(); rank++) {
		futs.push_back(
				hpx::async < transmit_chain_particles_action > (hpx_localities()[ranks[rank]], std::move(parts[rank])));
	}
	hpx::wait_all(futs.begin(), futs.end());
}

void chainmesh_create() {
	std::vector<hpx::future<void>> futs;
	for (auto c : hpx_children()) {
		futs.push_back(hpx::async < chainmesh_create_action > (c));
	}
	static int N = get_options().chain_dim;
	mybox = find_my_box(N);
	mybigbox = mybox.pad(1);
	cells.resize(mybigbox.volume());
	for (int i = 0; i < cells.size(); i++) {
		cells[i].pbegin = 0;
		cells[i].pend = 0;
	}
	thread_vol = std::max(1, (int) (cells.size() / hpx::thread::hardware_concurrency() / 8));
	futs.push_back(hpx::async(sort, mybigbox, 0, particles_local_size()));
	sort(mybigbox, particles_local_size(), particles_size());
	hpx::wait_all(futs.begin(), futs.end());
}

static void sort(const range<int> chain_box, int pbegin, int pend) {
	static int N = get_options().chain_dim;
	static double Ninv = 1.0 / N;
	const int vol = chain_box.volume();
	if (vol == 1) {
		if (pbegin != pend) {
			auto& cell = cells[mybigbox.index(chain_box.begin)];
			cell.pbegin = pbegin;
			cell.pend = pend;
		}
	} else {
		int long_dim;
		int long_span = -1;
		for (int dim = 0; dim < NDIM; dim++) {
			const int span = chain_box.end[dim] - chain_box.begin[dim];
			if (span > long_span) {
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
		if (vol > thread_vol) {
			auto futl = hpx::async(sort, chain_box_left, pbegin, pmid);
			auto futr = hpx::async(sort, chain_box_right, pmid, pend);
			futl.get();
			futr.get();
		} else {
			sort(chain_box_left, pbegin, pmid);
			sort(chain_box_right, pmid, pend);
		}
	}
}
