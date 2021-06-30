
#include <tigerpm/tigerpm.hpp>

#include <vector>

void hpx_init();
int hpx_rank();
int hpx_size();
const vector<hpx::id_type>& hpx_localities();
const vector<hpx::id_type>& hpx_children();
