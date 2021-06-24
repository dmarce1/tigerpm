#include <tigerpm/hpx.hpp>


static int rank;
static int nranks;
static std::vector<hpx::id_type> localities;
static std::vector<hpx::id_type> children;

#define CHILD_SHIFT 3

#define NCHILD_RANKS (1 << CHILD_SHIFT)


void hpx_init() {
	rank = hpx::get_locality_id();
	localities = hpx::find_all_localities();
	nranks = localities.size();
	int base = (rank - 1) << CHILD_SHIFT;
	for( int i = 0; i < NCHILD_RANKS; i++) {
		const int index = base + i + 1;
		if( index < nranks) {
			children.push_back(localities[index]);
		}
	}
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
