#include <tigerpm/kick_pme.hpp>
#include <tigerpm/chainmesh.hpp>
#include <tigerpm/particles.hpp>
#include <tigerpm/hpx.hpp>

HPX_PLAIN_ACTION(kick_pme_begin);
HPX_PLAIN_ACTION(kick_pme_end);

void kick_pme_begin(int min_rung, double scale, double t0, bool first_call) {
	vector<hpx::future<void>> futs;
	for (auto c : hpx_children()) {
		futs.push_back(hpx::async < kick_pme_begin_action > (c, min_rung, scale, t0, first_call));
	}

	kick_pme(chainmesh_interior_box(), min_rung, scale, t0, first_call);

	hpx::wait_all(futs.begin(),futs.end());
}

void kick_pme_end() {
	vector<hpx::future<void>> futs;
	for (auto c : hpx_children()) {
		futs.push_back(hpx::async < kick_pme_end_action > (c));
	}

	particles_resize(particles_local_size());

	hpx::wait_all(futs.begin(),futs.end());
}
