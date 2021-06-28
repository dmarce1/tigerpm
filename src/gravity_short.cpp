#include <tigerpm/gravity_short.hpp>
#include <tigerpm/hpx.hpp>
#include <tigerpm/particles.hpp>

using return_type = std::pair<std::vector<double>, std::array<std::vector<double>, NDIM>>;

static return_type do_ewald(const std::vector<fixed32>& sinkx, const std::vector<fixed32>& sinky,
		const std::vector<fixed32>& sinkz);

HPX_PLAIN_ACTION (do_ewald);

using return_type = std::pair<std::vector<double>, std::array<std::vector<double>, NDIM>>;

static return_type do_ewald(const std::vector<fixed32>& sinkx, const std::vector<fixed32>& sinky,
		const std::vector<fixed32>& sinkz) {
	std::vector < hpx::future < return_type >> futs;
	for (auto c : hpx_children()) {
		futs.push_back(hpx::async < do_ewald_action > (c, sinkx, sinky, sinkz));
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
	std::vector<fixed32> sinkx(Nsamples);
	std::vector<fixed32> sinky(Nsamples);
	std::vector<fixed32> sinkz(Nsamples);
	for (int i = 0; i < Nsamples; i++) {
		sinkx[i] = samples[i].x[XDIM];
		sinky[i] = samples[i].x[YDIM];
		sinkz[i] = samples[i].x[ZDIM];
	}
	auto results = do_ewald(sinkx, sinky, sinkz);
	for( int i = 0; i < Nsamples; i++) {
		printf( "%e\n", samples[i].g[1] / results.second[1][i]);
	}

}
