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

	auto samples = particles_sample(Nsamples);
	vector<fixed32> sinkx(Nsamples);
	vector<fixed32> sinky(Nsamples);
	vector<fixed32> sinkz(Nsamples);
	for (int i = 0; i < Nsamples; i++) {
		sinkx[i] = samples[i].x[XDIM];
		sinky[i] = samples[i].x[YDIM];
		sinkz[i] = samples[i].x[ZDIM];
	}
	double l2sum = 0.0;
	double l2norm = 0.0;
	auto results = do_ewald(sinkx, sinky, sinkz);
	for (int i = 0; i < Nsamples; i++) {
		double f1 = 0.0, f2 = 0.0;
		double r2 = 0.0;
		for (int dim = 0; dim < NDIM; dim++) {
			f1 += sqr(samples[i].g[dim]);
			r2 += sqr(samples[i].x[dim].to_double() - 0.5);
			f2 += sqr(results.second[dim][i]);
		}
		const double r = sqrt(r2);
		f1 = sqrt(f1);
		f2 = sqrt(f2);
		l2sum += sqr(f1-f2);
		l2norm += sqr(f2);
		printf("%e %e %e \n", f1, f2, f2 / f1);
	}
	l2sum = sqrt(l2sum/l2norm);
	PRINT( "L2 Error = %e\n", l2sum);
	 /*
	vector<fixed32> sinkx(Nsamples);
	vector<fixed32> sinky(Nsamples);
	vector<fixed32> sinkz(Nsamples);
	for (int i = 0; i < Nsamples; i++) {
		sinkx[i] = double(i) / Nsamples + 0.5 / Nsamples;
		sinky[i] = 0.5;
		sinkz[i] = 0.5;
	}
	FILE* fp = fopen("out.dat", "wt");
	auto results = do_ewald(sinkx, sinky, sinkz);
	double l1sum = 0.0, l2sum = 0.0;
	double l1norm = 0.0, l2norm = 0.0;
	for (int i = 0; i < Nsamples; i++) {
		if( i >= Nsamples /4 && i < 3* Nsamples/4) {
			continue;
		}
		array<double, NDIM> x;
		x[0] = sinkx[i].to_double();
		x[1] = sinky[i].to_double();
		x[2] = sinkz[i].to_double();
		auto g = gravity_long_force_at(x);
		double f1 = 0.0, f2 = 0.0;
		for (int dim = 0; dim < NDIM; dim++) {
			f1 += sqr(results.second[dim][i]);
			f2 += sqr(g.second[dim]);
		}
		l1sum += std::abs(f1 - f2);
		l1norm += std::abs(f1);
		l2sum += sqr(f1 - f2);
		l2norm += sqr(f1);
		fprintf(fp, "%e %e %e\n", x[0], results.second[0][i], g.second[0]);
	}
	fclose(fp);
	printf("L1 = %e \n", l1sum / l1norm);
	printf("L2 = %e \n", std::sqrt(l2sum / l2norm));*/
}
