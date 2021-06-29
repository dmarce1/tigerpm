#include <tigerpm/chainmesh.hpp>
#include <tigerpm/particles.hpp>
#include <tigerpm/util.hpp>

static std::vector<chaincell> cells;
static range<int> mybox;
static int thread_vol;

static void sort(const range<int> chain_box, int pbegin, int pend);

void chainmesh_create() {
	static int N = get_options().chain_dim;
	mybox = find_my_box(N);
	cells.resize(mybox.volume());
	thread_vol = std::max(1, (int) (cells.size() / hpx::thread::hardware_concurrency() / 8));
	sort(mybox, 0, particles_size());
}

static void sort(const range<int> chain_box, int pbegin, int pend) {
	static int N = get_options().chain_dim;
	static double Ninv = 1.0 / N;
	const int vol = chain_box.volume();
	if (vol == 1) {
		auto& cell = cells[mybox.index(chain_box.begin)];
		cell.pbegin = pbegin;
		cell.pend = pend;
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
