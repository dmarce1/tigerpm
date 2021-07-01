#include <tigerpm/kick_pme.hpp>
#include <tigerpm/chainmesh.hpp>

void kick_pme(int min_rung, double scale, double t0, bool first_call) {
	kick_pme(chainmesh_interior_box(), min_rung, scale, t0, first_call);
}
