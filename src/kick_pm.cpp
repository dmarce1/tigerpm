#include <tigerpm/gravity_long.hpp>
#include <tigerpm/kick_pm.hpp>
#include <tigerpm/hpx.hpp>
#include <tigerpm/particles.hpp>
#include <tigerpm/tigerpm.hpp>

HPX_PLAIN_ACTION (kick_pm);

void kick_pm() {
	std::vector<hpx::future<void>> futs;
	for (auto c : hpx_children()) {
		futs.push_back(hpx::async < kick_pm_action > (c));
	}

	for (int i = 0; i < particles_size(); i++) {
		std::array<double, NDIM> pos;
		for (int dim = 0; dim < NDIM; dim++) {
			pos[dim] = particles_pos(dim, i).to_double();
		}
		const auto gforce = gravity_long_force_at(pos);
//		printf( "%e %e %e\n", gforce.second[0],gforce.second[1],gforce.second[2]);
#ifdef FORCE_TEST
		for (int dim = 0; dim < NDIM; dim++) {
			particles_gforce(dim, i) = gforce.second[dim];
		}
		particles_pot(i) = gforce.first;
#endif
	}

	hpx::wait_all(futs.begin(), futs.end());
}
