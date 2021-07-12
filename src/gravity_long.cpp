#include <tigerpm/fft.hpp>
#include <tigerpm/gravity_long.hpp>
#include <tigerpm/hpx.hpp>
#include <tigerpm/options.hpp>
#include <tigerpm/particles.hpp>
#include <tigerpm/util.hpp>

static vector<float> phi;
static range<int> source_box;

void compute_source();
void apply_laplacian(gravity_long_type);
void get_phi();

HPX_PLAIN_ACTION(compute_source);
HPX_PLAIN_ACTION(get_phi);
HPX_PLAIN_ACTION(apply_laplacian);

range<int> gravity_long_get_phi_box() {
	return source_box;
}

vector<float> gravity_long_get_phi(const range<int>& this_box) {
	vector<float> this_phi(this_box.volume());
	array<int, NDIM> i;
	for (i[0] = this_box.begin[0]; i[0] != this_box.end[0]; i[0]++) {
		for (i[1] = this_box.begin[1]; i[1] != this_box.end[1]; i[1]++) {
			for (i[2] = this_box.begin[2]; i[2] != this_box.end[2]; i[2]++) {
				this_phi[this_box.index(i)] = phi[source_box.index(i)];
			}
		}
	}
	return this_phi;
}

void gravity_long_compute(gravity_long_type type) {
	const double N = get_options().four_dim;
	fft3d_init(N);
	PRINT("Computing source\n");
	compute_source();
	fft3d_execute();
	PRINT("Apply LaPlacian\n");
	apply_laplacian(type);
	fft3d_inv_execute();
	PRINT("get phi\n");
	get_phi();
	fft3d_destroy();
}

#define THIS_NINTERP 4

std::pair<float, array<float, NDIM>> gravity_long_force_at(const array<double, NDIM>& pos) {
	double phi0;
	array<double, NDIM> g;
	array<float, NDIM> gret;
	array<int, NDIM> I;
	array<double, NDIM> X;
	array<array<double, THIS_NINTERP>, NDIM> w;
	array<array<double, THIS_NINTERP>, NDIM> dw;
	const double N = get_options().four_dim;

	for (int dim = 0; dim < NDIM; dim++) {
		X[dim] = pos[dim] * N;
		I[dim] = int(X[dim]);
		X[dim] -= double(I[dim]);
		I[dim] -= 1;
	}
	for (int dim = 0; dim < NDIM; dim++) {
		double x1 = X[dim];
		const double x2 = X[dim] * x1;
		const double x3 = x1 * x2;
		w[dim][0] = -0.5 * x1 + x2 - 0.5 * x3;
		w[dim][1] = 1.0 - 2.5 * x2 + 1.5 * x3;
		w[dim][2] = 0.5 * x1 + 2.0 * x2 - 1.5 * x3;
		w[dim][3] = -0.5 * x2 + 0.5 * x3;
		dw[dim][0] = -0.5 + 2.0 * x1 - 1.5 * x2;
		dw[dim][1] = -5.0 * x1 + 4.5 * x2;
		dw[dim][2] = 0.5 + 4.0 * x1 - 4.5 * x2;
		dw[dim][3] = -x1 + 1.5 * x2;
	}
	array<int, NDIM> J;
	for (int dim1 = 0; dim1 < NDIM; dim1++) {
		g[dim1] = 0.0;
		for (J[0] = I[0]; J[0] < I[0] + THIS_NINTERP; J[0]++) {
			for (J[1] = I[1]; J[1] < I[1] + THIS_NINTERP; J[1]++) {
				for (J[2] = I[2]; J[2] < I[2] + THIS_NINTERP; J[2]++) {
					double w0 = 1.0;
					for (int dim2 = 0; dim2 < NDIM; dim2++) {
						const int i0 = J[dim2] - I[dim2];
						if (dim1 == dim2) {
							w0 *= dw[dim2][i0];
						} else {
							w0 *= w[dim2][i0];
						}
					}
					const int l = source_box.index(J);
					assert(l >= 0);
					assert(l < phi.size());
					g[dim1] += w0 * phi[l] * N;
				}
			}
		}

	}
	phi0 = 0.0;
	for (J[0] = I[0]; J[0] < I[0] + THIS_NINTERP; J[0]++) {
		for (J[1] = I[1]; J[1] < I[1] + THIS_NINTERP; J[1]++) {
			for (J[2] = I[2]; J[2] < I[2] + THIS_NINTERP; J[2]++) {
				double w0 = 1.0;
				for (int dim2 = 0; dim2 < NDIM; dim2++) {
					const int i0 = J[dim2] - I[dim2];
					w0 *= w[dim2][i0];
				}
				const int l = source_box.index(J);
				assert(l >= 0);
				assert(l < phi.size());
				phi0 += w0 * phi[l];
			}
		}
	}
	for (int dim = 0; dim < NDIM; dim++) {
		gret[dim] = -g[dim];
	}
	return std::make_pair((float) phi0, gret);
}


void compute_source() {
	vector<hpx::future<void>> futs;
	for (auto c : hpx_children()) {
		futs.push_back(hpx::async<compute_source_action>(c));
	}
	auto source = gravity_long_compute_source_local();

	source_box = find_my_box(get_options().chain_dim);
	for (int dim = 0; dim < NDIM; dim++) {
		const static auto ratio = get_options().four_o_chain;
		source_box.begin[dim] *= ratio;
		source_box.end[dim] *= ratio;
	}
	source_box = source_box.pad(PHI_BW);
	source.resize(source_box.volume(), 0.0f);
	/*const double N = get_options().four_dim;
	const int xdim = source_box.end[XDIM] - source_box.begin[XDIM];
	mutexes.resize(xdim);
	for (int i = 0; i < xdim; i++) {
		mutexes[i] = std::make_shared<spinlock_type>();
	}
	const int nthreads = hpx::thread::hardware_concurrency();
	for (int proc = 0; proc < nthreads; proc++) {
		futs.push_back(hpx::async([proc,nthreads,N,&source,&mutexes]() {
			const int begin = size_t(proc) * size_t(particles_size()) / size_t(nthreads);
			const int end = size_t(proc+1) * size_t(particles_size()) / size_t(nthreads);
			for (int i = begin; i < end; i++) {
				array<double, NDIM> X;
				array<int, NDIM> I;
				array<array<double,CLOUD_W>,NDIM> w;
				X[XDIM] = particles_pos(0, i).to_double();
				X[YDIM] = particles_pos(1, i).to_double();
				X[ZDIM] = particles_pos(2, i).to_double();
				for( int dim = 0; dim < NDIM; dim++) {
					I[dim] = int(X[dim] * N + PHI_BW - 1) - PHI_BW;
					for( int i = 0; i < CLOUD_W; i++) {
						w[dim][i] = cloud4(X[dim] * N - I[dim] - i);
//						PRINT( "%e\n", X[dim] * N - I[dim] - i);
					}
				}
				array<int, NDIM> J;
				const double c0 = 4.0 * M_PI * N;
				for( J[0] = 0; J[0] < CLOUD_W; J[0]++) {
					std::lock_guard<spinlock_type> lock(*mutexes[I[0] + J[0] - source_box.begin[XDIM]]);
					for( J[1] = 0; J[1] < CLOUD_W; J[1]++) {
						for( J[2] = 0; J[2] < CLOUD_W; J[2]++) {
							source[source_box.index(I[0]+J[0],I[1]+J[1],I[2]+J[2])] += c0 * w[0][J[0]] * w[1][J[1]] * w[2][J[2]];
						}
					}
				}
			}
		}));
	}*/

	hpx::wait_all(futs.begin(), futs.end());
	fft3d_accumulate_real(source_box, std::move(source));
}

void apply_laplacian(gravity_long_type type) {
	vector<hpx::future<void>> futs;
	for (auto c : hpx_children()) {
		futs.push_back(hpx::async<apply_laplacian_action>(c, type));
	}
	const double N = get_options().four_dim;
	const double rs2 = sqr(get_options().rs * N);
	const auto box = fft3d_complex_range();
	array<int, NDIM> i;
	array<double, NDIM> k;
	auto& Y = fft3d_complex_vector();
	const double c0 = 2.0 * M_PI / N;
	const double dx = 1.0;
	for (i[0] = box.begin[0]; i[0] < box.end[0]; i[0]++) {
		for (i[1] = box.begin[1]; i[1] < box.end[1]; i[1]++) {
			for (i[2] = box.begin[2]; i[2] < box.end[2]; i[2]++) {
				for (int dim = 0; dim < NDIM; dim++) {
					k[dim] = i[dim] < N / 2 ? i[dim] : i[dim] - N;
					k[dim] *= c0;
				}
				//			const double cosnk2 = 2.0 * (cos(k[0]) + cos(k[1]) + cos(k[2]) - 3.0);
				const double nk2 = -sqr(k[0], k[1], k[2]);
				const int index = box.index(i);
				if (nk2 < 0.0) {
					const double nk2inv = 1.0 / nk2;
					Y[index] *= float(nk2inv);
				} else {
					Y[index] *= 0.0;
				}
				if (type == GRAVITY_LONG_PME) {
					Y[index] *= exp(nk2 * rs2);
				}

				const double sx = sinc(0.5 * k[0] * dx);
				const double sy = sinc(0.5 * k[1] * dx);
				const double sz = sinc(0.5 * k[2] * dx);
				const double s = sx * sy * sz;
				Y[index] *= std::pow(s,-8);
			}
		}
	}

	hpx::wait_all(futs.begin(), futs.end());
}

void get_phi() {
	vector<hpx::future<void>> futs;
	for (auto c : hpx_children()) {
		futs.push_back(hpx::async<get_phi_action>(c));
	}

	phi = fft3d_read_real(source_box);

	hpx::wait_all(futs.begin(), futs.end());
}
