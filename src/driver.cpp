#include <tigerpm/drift.hpp>
#include <tigerpm/fmmpm.hpp>
#include <tigerpm/initialize.hpp>
#include <tigerpm/gravity_long.hpp>
#include <tigerpm/particles.hpp>
#include <tigerpm/timer.hpp>

using rung_type = std::int8_t;
using time_type = std::uint64_t;

inline time_type inc(time_type t, rung_type max_rung) {
	t += (time_type(1) << time_type(64 - max_rung));
	return t;
}

inline rung_type min_rung(time_type t) {
	rung_type min_rung = 64;
	while (((t & 1) == 0) && (min_rung != 0)) {
		min_rung--;
		t >>= 1;
	}
	return min_rung;
}

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

kick_return kick_step(int minrung, double scale, double t0, double theta, bool first_call, bool full_eval) {
	particles_domain_sort();
	gravity_long_compute(GRAVITY_LONG_PME);
	chainmesh_create();
	chainmesh_exchange_begin();
	chainmesh_exchange_end();
	kick_return kr = kick_fmmpm_begin(minrung, scale, t0, theta, first_call, full_eval);
	kick_fmmpm_end();
	return kr;
}

struct driver_params {
	double a;
	double tau;
	double tau_max;
	double cosmicK;
	double esum0;
	int iter;
	time_type itime;
};

void write_checkpoint(driver_params params) {
	PRINT("Writing checkpoint\n");
	const std::string fname = std::string("checkpoint.") + std::to_string(params.iter) + std::string(".dat");
	FILE* fp = fopen(fname.c_str(), "wb");
	if (fp == nullptr) {
		PRINT("Unable to open %s for writing.\n", fname.c_str());
		abort();
	}
	fwrite(&params, sizeof(driver_params), 1, fp);
	particles_save(fp);

	fclose(fp);
}

void read_checkpoint(driver_params& params, int checknum) {
	PRINT("Reading checkpoint\n");
	const std::string fname = std::string("checkpoint.") + std::to_string(checknum) + std::string(".dat");
	FILE* fp = fopen(fname.c_str(), "rb");
	if (fp == nullptr) {
		PRINT("Unable to open %s for reading.\n", fname.c_str());
		abort();
	}
	FREAD(&params, sizeof(driver_params), 1, fp);
	particles_load(fp);

	fclose(fp);
}

void driver() {
	driver_params params;
	double a0 = 1.0 / (1.0 + get_options().z0);
	if (get_options().check_num >= 0) {
		read_checkpoint(params, get_options().check_num);
	} else {
		initialize();
		params.tau_max = cosmos_age(a0);
		params.tau = 0.0;
		params.a = a0;
		params.cosmicK = 0.0;
		params.itime = 0;
		params.iter = 0;
	}
	auto& a = params.a;
	auto& tau = params.tau;
	auto& tau_max = params.tau_max;
	auto& cosmicK = params.cosmicK;
	auto& esum0 = params.esum0;
	auto& itime = params.itime;
	auto& iter = params.iter;
	double t0 = tau_max / 100.0;
	double pot;
	timer tmr;
	tmr.start();
	while (tau < tau_max) {
		tmr.stop();
		if (tmr.read() > get_options().check_freq) {
			write_checkpoint(params);
			tmr.reset();
		}
		tmr.start();
		int minrung = min_rung(itime);
		bool full_eval = minrung == 0;
		double theta;
		const double z = 1.0 / a - 1.0;
		if (z > 20.0) {
			theta = 1.0 / 3.0;
		} else if (z > 2.0) {
			theta = 1.0 / 2.0;
		} else {
			theta = 2.0 / 3.0;
		}
		kick_return kr = kick_step(minrung, a, t0, theta, tau == 0.0, full_eval);
		if (full_eval) {
			pot = kr.pot * 0.5 / a;
		}
		double dt = t0 / (1 << kr.max_rung);
		const double dadt1 = cosmos_dadtau(a);
		const double a1 = a;
		a += dadt1 * dt;
		const double dadt2 = cosmos_dadtau(a);
		a += 0.5 * (dadt2 - dadt1) * dt;
		const double a2 = 2.0 / (1.0 / a + 1.0 / a1);
		drift_return dr = drift(a2, dt);
		cosmicK += dr.kin * (a - a1);
		const double esum = (a * (pot + dr.kin) + cosmicK);
		if (tau == 0.0) {
			esum0 = esum;
		}
		const double eerr = (esum - esum0) / (a * dr.kin + a * std::abs(pot) + cosmicK);
		if (full_eval) {
			PRINT("\n%12s %12s %12s %12s %12s %12s %12s %12s %12s %12s %12s\n", "i", "Z", "time", "dt", "pot", "kin", "cosmicK", "pot err", "min rung", "max rung",
					"nactive");
		}
		PRINT("%12i %12.3e %12.3e %12.3e %12.3e %12.3e %12.3e %12.3e %12i %12i %12i\n", iter, z, tau / tau_max, dt / tau_max, a * pot, a * dr.kin, cosmicK, eerr,
				minrung, kr.max_rung, kr.nactive);

		itime = inc(itime, kr.max_rung);
		tau += dt;
		iter++;
	}

}
