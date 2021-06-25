#include <tigerpm/fft.hpp>
#include <tigerpm/gravity_long.hpp>
#include <tigerpm/hpx.hpp>
#include <tigerpm/options.hpp>
#include <tigerpm/particles.hpp>

static std::vector<float> phi;
static range<int> source_box;

void compute_source();
void apply_laplacian();
void get_phi();

HPX_PLAIN_ACTION (compute_source);
HPX_PLAIN_ACTION (get_phi);
HPX_PLAIN_ACTION (apply_laplacian);

void gravity_long_compute() {
	const double N = get_options().part_dim;
	fft3d_init(N);
	compute_source();
	fft3d_execute();
	/* apply laplacian */
	fft3d_inv_execute();
	get_phi;
	fft3d_destroy();
}

void compute_source() {
	std::vector<hpx::future<void>> futs;
	for (auto c : hpx_children()) {
		futs.push_back(hpx::async < compute_source_action > (c));
	}
	std::vector<float> source;
	std::vector < std::shared_ptr < spinlock_type >> mutexes;

	source_box = particles_get_local_box().pad(1);
	source.resize(source_box.volume(), 0.0f);
	const double N = get_options().part_dim;
	const int xdim = source_box.end[XDIM] - source_box.begin[XDIM];
	mutexes.resize(xdim);
	for (int i = 0; i < xdim; i++) {
		mutexes[i] = std::make_shared<spinlock_type>();
	}
	const int nthreads = hpx::threads::hardware_concurrency();
	for (int proc = 0; proc < nthreads; proc++) {
		futs.push_back(hpx::async([proc,nthreads,N,&source,&mutexes]() {
			const int begin = size_t(proc) * size_t(particles_size()) / size_t(nthreads);
			const int end = size_t(proc+1) * size_t(particles_size()) / size_t(nthreads);
			for (int i = begin; i < end; i++) {
				const double x = particles_pos(0, i).to_double();
				const double y = particles_pos(1, i).to_double();
				const double z = particles_pos(2, i).to_double();
				const int i0 = x * N;
				const int j0 = y * N;
				const int k0 = z * N;
				const int i1 = i0 + 1;
				const int j1 = j0 + 1;
				const int k1 = k0 + 1;
				const double w1x = x * N - i0;
				const double w1y = y * N - j0;
				const double w1z = z * N - k0;
				const double w0x = 1.0 - w1x;
				const double w0y = 1.0 - w1y;
				const double w0z = 1.0 - w1z;
				const double w1yw1z = w1y * w1z;
				const double w0yw1z = w0y * w1z;
				const double w1yw0z = w1y * w0z;
				const double w0yw0z = w0y * w0z;
				{
					std::lock_guard<spinlock_type> lock(*mutexes[i0 - source_box.begin[XDIM]]);
					source[source_box.index(i0, j0, k0)] += w0x * w0yw0z;
					source[source_box.index(i0, j0, k1)] += w0x * w0yw1z;
					source[source_box.index(i0, j1, k0)] += w0x * w1yw0z;
					source[source_box.index(i0, j1, k1)] += w0x * w1yw1z;
				}
				{
					std::lock_guard<spinlock_type> lock(*mutexes[i1 - source_box.begin[XDIM]]);
					source[source_box.index(i1, j0, k0)] += w1x * w0yw0z;
					source[source_box.index(i1, j0, k1)] += w1x * w0yw1z;
					source[source_box.index(i1, j1, k0)] += w1x * w1yw0z;
					source[source_box.index(i1, j1, k1)] += w1x * w1yw1z;
				}
			}
		}));
	}

	hpx::wait_all(futs.begin(), futs.end());
	fft3d_accumulate_real(source_box, std::move(source));
}

void apply_laplacian() {
	std::vector<hpx::future<void>> futs;
	for (auto c : hpx_children()) {
		futs.push_back(hpx::async < apply_laplacian_action > (c));
	}
	const double N = get_options().part_dim;
	const auto box = fft3d_complex_range();
	std::array<int, NDIM> i;
	std::array<double, NDIM> k;
	auto& Y = fft3d_complex_vector();
	const double c0 = 2.0 * M_PI / get_options().box_size;
	for (i[0] = box.begin[0]; i[0] < box.end[0]; i[0]++) {
		for (i[1] = box.begin[1]; i[1] < box.end[1]; i[1]++) {
			for (i[2] = box.begin[2]; i[2] < box.end[2]; i[2]++) {
				for (int dim = 0; dim < NDIM; dim++) {
					k[dim] = i[dim] < N / 2 ? i[dim] : i[dim] - N;
					k[dim] *= c0;
				}
				const double k2 = k[0] * k[0] + k[1] * k[1] + k[2] * k[2];
				const double k2inv = 1.0 / k2;
				Y[box.index(i)] *= float(-k2inv);
			}
		}
	}

	hpx::wait_all(futs.begin(), futs.end());
}

void get_phi() {
	std::vector<hpx::future<void>> futs;
	for (auto c : hpx_children()) {
		futs.push_back(hpx::async < get_phi_action > (c));
	}

	phi = fft3d_read_real(source_box);

	hpx::wait_all(futs.begin(), futs.end());
}
