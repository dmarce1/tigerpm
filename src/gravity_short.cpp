#include <tigerpm/gravity_short.hpp>
#include <tigerpm/gravity_long.hpp>
#include <tigerpm/hpx.hpp>
#include <tigerpm/particles.hpp>
#include <tigerpm/util.hpp>

using return_type = std::pair<vector<double>, array<vector<double>, NDIM>>;

static return_type do_ewald(const vector<fixed32>& sinkx, const vector<fixed32>& sinky, const vector<fixed32>& sinkz);

HPX_PLAIN_ACTION(do_ewald);

using return_type = std::pair<vector<double>, array<vector<double>, NDIM>>;

static return_type do_ewald(const vector<fixed32>& sinkx, const vector<fixed32>& sinky, const vector<fixed32>& sinkz) {
	vector<hpx::future<return_type>> futs;
	for (auto c : hpx_children()) {
		futs.push_back(hpx::async<do_ewald_action>(c, sinkx, sinky, sinkz));
	}

	auto results = gravity_short_ewald_call_kernel(sinkx, sinky, sinkz);

	for (auto& f : futs) {
		auto other = f.get();
		for (int i = 0; i < sinkx.size(); i++) {
			results.first[i] += other.first[i];
			for (int dim = 0; dim < NDIM; dim++) {
				results.second[dim][i] += other.second[dim][i];
			}
		}
	}
	return results;
}

void gravity_short_ewald_compare(int Nsamples) {
#ifdef FORCE_TEST
	auto samples = particles_sample(Nsamples);
	vector<fixed32> sinkx(Nsamples);
	vector<fixed32> sinky(Nsamples);
	vector<fixed32> sinkz(Nsamples);
	for (int i = 0; i < Nsamples; i++) {
		sinkx[i] = samples[i].x[XDIM];
		sinky[i] = samples[i].x[YDIM];
		sinkz[i] = samples[i].x[ZDIM];
	}
	double l2sum_phi = 0.0;
	double l2norm_phi = 0.0;
	double l2sum_force = 0.0;
	double l2norm_force = 0.0;
	auto results = do_ewald(sinkx, sinky, sinkz);
	double lmax_phi = 0.0;
	double lmax_force = 0.0;
	for (int i = 0; i < Nsamples; i++) {
		double f1 = 0.0, f2 = 0.0;
		double g1 = 0.0, g2 = 0.0;
		for (int dim = 0; dim < NDIM; dim++) {
			g1 += sqr(samples[i].g[dim]);
			g2 += sqr(results.second[dim][i]);
		}
		g1 = sqrt(g1);
		g2 = sqrt(g2);
		f1 = samples[i].p;
		f2 = results.first[i];
		l2sum_phi += sqr(f1 - f2);
		lmax_phi = std::max(lmax_phi, std::abs(f1 - f2));
		l2norm_phi += sqr(f2);
		l2sum_force += sqr(g1 - g2);
		lmax_force = std::max(lmax_force, std::abs(g1 - g2));
		l2norm_force += sqr(g2);
		printf("%e %e %e | %e %e %e \n", sinkx[i].to_float(), sinky[i].to_float(), sinkz[i].to_float(), g1, g2, g2 / g1);
	}
	l2sum_force = sqrt(l2sum_force / l2norm_force);
	lmax_force /= sqrt(l2norm_force) / Nsamples;
	PRINT("Force RMS Error     = %e\n", l2sum_force);
	PRINT("Force Max Error     = %e\n", lmax_force);
	l2sum_phi = sqrt(l2sum_phi / l2norm_phi);
	lmax_phi /= sqrt(l2norm_phi) / Nsamples;
	PRINT("Potential RMS Error = %e\n", l2sum_phi);
	PRINT("Potential Max Error = %e\n", lmax_phi);

#endif
}
