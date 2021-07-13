#include <tigerpm/drift.hpp>
#include <tigerpm/fmmpm.hpp>
#include <tigerpm/initialize.hpp>
#include <tigerpm/gravity_long.hpp>
#include <tigerpm/particles.hpp>

double cosmos_dadtau(double a) {
	const auto H = constants::H0 * get_options().code_to_s * get_options().hubble;
	const auto omega_m = get_options().omega_m;
	const auto omega_r = get_options().omega_r;
	const auto omega_lambda = 1.0 - omega_m - omega_r;
	return H * a * a * std::sqrt(omega_r / (a * a * a * a) + omega_m / (a * a * a) + omega_lambda);
}


double cosmos_age(double a0) {
	double a = a0;
	double t = 0.0;
	while (a < 1.0) {
		const double dadt1 = cosmos_dadtau(a);
		const double dt = (a / dadt1) * 1.e-5;
		const double dadt2 = cosmos_dadtau(a + dadt1 * dt);
		a += 0.5 * (dadt1 + dadt2) * dt;
		t += dt;
	}
	return t;
}

kick_return kick_step(int minrung, double scale, double dt, bool first_call) {
	particles_domain_sort();
	gravity_long_compute(GRAVITY_LONG_PME);
	chainmesh_create();
	chainmesh_exchange_begin();
	chainmesh_exchange_end();
	kick_return kr = kick_fmmpm_begin(minrung, scale, dt, first_call);
	kick_fmmpm_end();
	return kr;
}

void driver() {
	initialize();
	double a0 = 1.0 / (1.0 + get_options().z0);
	double tau_max = cosmos_age(a0);

}
