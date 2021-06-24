#include <tigerpm/hpx.hpp>

static int rank;
static int nranks;
static std::vector<hpx::id_type> localities;
static std::vector<hpx::id_type> children;

HPX_PLAIN_ACTION (hpx_init);

void hpx_init() {
	rank = hpx::get_locality_id();
	localities = hpx::find_all_localities();
	nranks = localities.size();
	int base = (rank + 1) << 1;
	const int index1 = base - 1;
	const int index2 = base;
	if (index1 < nranks) {
		children.push_back(localities[index1]);
	}
	if (index2 < nranks) {
		children.push_back(localities[index2]);
	}

	std::vector<hpx::future<void>> futs;
	for (auto c : hpx_children()) {
		futs.push_back(hpx::async < hpx_init_action > (c));
	}
	hpx::wait_all(futs.begin(), futs.end());

}

int hpx_rank() {
	return rank;
}

int hpx_size() {
	return nranks;
}

const std::vector<hpx::id_type>& hpx_localities() {
	return localities;
}

const std::vector<hpx::id_type>& hpx_children() {
	return children;
}
