#include <tigerpm/kick_pme.hpp>
#include <tigerpm/chainmesh.hpp>
#include <tigerpm/particles.hpp>
#include <tigerpm/gravity_short.hpp>
#include <tigerpm/gravity_long.hpp>
#include <tigerpm/util.hpp>
#include <tigerpm/timer.hpp>
#include <algorithm>

#define MAX_RUNG 32
#define NINTERP 6
#define NCELLS 27
#define KICK_PME_BLOCK_SIZE 1024

__constant__ float rung_dt[MAX_RUNG] = { 1.0 / (1 << 0), 1.0 / (1 << 1), 1.0 / (1 << 2), 1.0 / (1 << 3), 1.0 / (1 << 4), 1.0 / (1 << 5), 1.0 / (1 << 6), 1.0
		/ (1 << 7), 1.0 / (1 << 8), 1.0 / (1 << 9), 1.0 / (1 << 10), 1.0 / (1 << 11), 1.0 / (1 << 12), 1.0 / (1 << 13), 1.0 / (1 << 14), 1.0 / (1 << 15), 1.0
		/ (1 << 16), 1.0 / (1 << 17), 1.0 / (1 << 18), 1.0 / (1 << 19), 1.0 / (1 << 20), 1.0 / (1 << 21), 1.0 / (1 << 22), 1.0 / (1 << 23), 1.0 / (1 << 24), 1.0
		/ (1 << 25), 1.0 / (1 << 26), 1.0 / (1 << 27), 1.0 / (1 << 28), 1.0 / (1 << 29), 1.0 / (1 << 30), 1.0 / (1 << 31) };

struct shmem_type {
	array<fixed32, KICK_PME_BLOCK_SIZE> x;
	array<fixed32, KICK_PME_BLOCK_SIZE> y;
	array<fixed32, KICK_PME_BLOCK_SIZE> z;
	array<int, KICK_PME_BLOCK_SIZE> index;
};

struct source_cell {
	int begin;
	int end;
};

struct sink_cell {
	int begin;
	int end;
	array<int, NDIM> loc;
};

struct kernel_params {
	fixed32* x;
	fixed32* y;
	fixed32* z;
	float* velx;
	float* vely;
	float* velz;
	float* phi;
	char* rung;
	source_cell* source_cells;
	sink_cell* sink_cells;
	int* active_sinki;
	int* active_sourcei;
	int nsink_cells;
	int min_rung;
	float rs;
	float GM;
	float eta;
	float t0;
	float scale;
	float hsoft;
	float inv2rs;
	float twooversqrtpi;
	float h2;
	float hinv;
	float h3inv;
	bool first_call;
	int Nfour;
	range<int> phi_box;
#ifdef FORCE_TEST
	float* gx;
	float* gy;
	float* gz;
	float* pot;
#endif
	void allocate(size_t source_size, size_t sink_size, size_t cell_count, size_t big_cell_count, size_t phi_cell_count) {
		nsink_cells = cell_count;
		CUDA_CHECK(cudaMalloc(&source_cells, cell_count * NCELLS * sizeof(source_cell)));
		CUDA_CHECK(cudaMalloc(&sink_cells, cell_count * sizeof(sink_cell)));
		CUDA_CHECK(cudaMalloc(&x, source_size * sizeof(fixed32)));
		CUDA_CHECK(cudaMalloc(&y, source_size * sizeof(fixed32)));
		CUDA_CHECK(cudaMalloc(&z, source_size * sizeof(fixed32)));
		CUDA_CHECK(cudaMalloc(&velx, sink_size * sizeof(float)));
		CUDA_CHECK(cudaMalloc(&vely, sink_size * sizeof(float)));
		CUDA_CHECK(cudaMalloc(&velz, sink_size * sizeof(float)));
		CUDA_CHECK(cudaMalloc(&rung, sink_size * sizeof(char)));
		CUDA_CHECK(cudaMalloc(&phi, phi_cell_count * sizeof(float)));
		CUDA_CHECK(cudaMalloc(&active_sinki, sink_size * sizeof(int)));
		CUDA_CHECK(cudaMalloc(&active_sourcei, sink_size * sizeof(int)));
#ifdef FORCE_TEST
		CUDA_CHECK(cudaMalloc(&gx, source_size * sizeof(float)));
		CUDA_CHECK(cudaMalloc(&gy, source_size * sizeof(float)));
		CUDA_CHECK(cudaMalloc(&gz, source_size * sizeof(float)));
		CUDA_CHECK(cudaMalloc(&pot, source_size * sizeof(float)));
#endif
	}
	void free() {
		CUDA_CHECK(cudaFree(x));
		CUDA_CHECK(cudaFree(y));
		CUDA_CHECK(cudaFree(z));
		CUDA_CHECK(cudaFree(velx));
		CUDA_CHECK(cudaFree(vely));
		CUDA_CHECK(cudaFree(velz));
		CUDA_CHECK(cudaFree(phi));
		CUDA_CHECK(cudaFree(active_sinki));
		CUDA_CHECK(cudaFree(active_sourcei));
		CUDA_CHECK(cudaFree(rung));
		CUDA_CHECK(cudaFree(source_cells));
		CUDA_CHECK(cudaFree(sink_cells));
#ifdef FORCE_TEST
		CUDA_CHECK(cudaFree(gx));
		CUDA_CHECK(cudaFree(gy));
		CUDA_CHECK(cudaFree(gz));
		CUDA_CHECK(cudaFree(pot));
#endif
	}
};

static size_t mem_requirements(int nsources, int nsinks, int vol, int bigvol, int phivol) {
	size_t mem = 0;
	mem += NDIM * sizeof(fixed32) * nsources;
	mem += NDIM * sizeof(float) * nsinks;
	mem += sizeof(char) * nsinks;
	mem += NCELLS * bigvol * sizeof(source_cell);
	mem += vol * sizeof(sink_cell);
	mem += 2 * sizeof(int) * vol;
	mem += phivol * sizeof(float);
	mem += sizeof(kernel_params);
#ifdef FORCE_TEST
	mem += (NDIM + 1) * sizeof(float) * nsinks;
#endif
	return mem;
}

__device__ inline float erfcexp(float x, float *e) {
	const float p(0.3275911f);
	const float a1(0.254829592f);
	const float a2(-0.284496736f);
	const float a3(1.421413741f);
	const float a4(-1.453152027f);
	const float a5(1.061405429f);
	const float t1 = 1.f / fmaf(p, x, 1.f);
	const float t2 = t1 * t1;
	const float t3 = t2 * t1;
	const float t4 = t2 * t2;
	const float t5 = t2 * t3;
	*e = expf(-x * x);
	return fmaf(a1, t1, fmaf(a2, t2, fmaf(a3, t3, fmaf(a4, t4, a5 * t5)))) * *e;
}

__constant__ kernel_params dev_params;

__global__ void kick_pme_kernel() {
	const kernel_params& params = dev_params;
	__shared__ shmem_type shmem;
	const int& tid = threadIdx.x;
	const int& bid = blockIdx.x;
	const int& gsz = gridDim.x;
	const float& inv2rs = params.inv2rs;
	const float& twooversqrtpi = params.twooversqrtpi;
	const float& h2 = params.h2;
	const float& hinv = params.hinv;
	const float& h3inv = params.h3inv;
	const int cell_begin = size_t(bid) * (size_t) params.nsink_cells / (size_t) gsz;
	const int cell_end = size_t(bid + 1) * (size_t) params.nsink_cells / (size_t) gsz;
	for (int cell_index = cell_begin; cell_index < cell_end; cell_index++) {
		const sink_cell& sink = params.sink_cells[cell_index];
		int* active_sinki = params.active_sinki + sink.begin;
		int* active_sourcei = params.active_sourcei + sink.begin;
		const auto& my_source_cell = params.source_cells[cell_index * NCELLS + NCELLS / 2];
		const int nsinks = sink.end - sink.begin;
		const int imax = round_up(nsinks, KICK_PME_BLOCK_SIZE);
		int nactive = 0;
		for (int i = tid; i < imax; i += KICK_PME_BLOCK_SIZE) {
			const int this_index = sink.begin + i;
			bool is_active;
			if (i < nsinks) {
				is_active = int(params.rung[this_index] >= params.min_rung);
			} else {
				is_active = false;
			}
			shmem.index[tid] = int(is_active);
			for (int P = 1; P < KICK_PME_BLOCK_SIZE; P *= 2) {
				int tmp;
				__syncthreads();
				if (tid >= P) {
					tmp = shmem.index[tid - P];
				}
				__syncthreads();
				if (tid >= P) {
					shmem.index[tid] += tmp;
				}
			}
			__syncthreads();
			int active_index = (tid > 0 ? shmem.index[tid - 1] : 0) + nactive;
			if (is_active) {
				active_sinki[active_index] = this_index;
				active_sourcei[active_index] = my_source_cell.begin + i;
			}
			nactive += shmem.index[KICK_PME_BLOCK_SIZE - 1];
			__syncthreads();
		}
		const int maxsink = round_up(nactive, KICK_PME_BLOCK_SIZE);
		for (int sink_index = tid; sink_index < maxsink; sink_index += KICK_PME_BLOCK_SIZE) {
			array<float, NDIM> g;
			g[0] = g[1] = g[2] = 0.f;
			float phi = 0.f;
			int srci;
			int snki;
			fixed32 sink_x;
			fixed32 sink_y;
			fixed32 sink_z;
			if (sink_index < nactive) {
				srci = active_sourcei[sink_index];
				snki = active_sinki[sink_index];
				sink_x = params.x[srci];
				sink_y = params.y[srci];
				sink_z = params.z[srci];
				array<int, NDIM> I;
				array<int, NDIM> J;
				array<float, NDIM> X;
				X[XDIM] = sink_x.to_float();
				X[YDIM] = sink_y.to_float();
				X[ZDIM] = sink_z.to_float();
				array<array<float, NINTERP>, NINTERP> w;
				array<array<float, NINTERP>, NINTERP> dw;
				for (int dim = 0; dim < NDIM; dim++) {
					X[dim] *= params.Nfour;
					I[dim] = min(int(X[dim]), params.phi_box.end[dim] - PHI_BW);
					X[dim] -= float(I[dim]);
					I[dim] -= 2;
				}
				for (int dim = 0; dim < NDIM; dim++) {
					float x1 = X[dim];
					float x2 = X[dim] * x1;
					float x3 = x1 * x2;
					float x4 = x2 * x2;
					float x5 = x3 * x2;
					w[dim][0] = (1.f / 12.f) * x1 - (1.f / 24.f) * x2 - (3.f / 8.f) * x3 + (13.f / 24.f) * x4 - (5.f / 24.f) * x5;
					w[dim][1] = -(2.f / 3.f) * x1 + (2.f / 3.f) * x2 + (13.f / 8.f) * x3 - (8.f / 3.f) * x4 + (25.f / 24.f) * x5;
					w[dim][2] = 1.0f - (5.f / 4.f) * x2 - (35.f / 12.f) * x3 + (21.f / 4.f) * x4 - (25.f / 12.f) * x5;
					w[dim][3] = (2.f / 3.f) * x1 + (2.f / 3.f) * x2 + (11.f / 4.f) * x3 - (31.f / 6.f) * x4 + (25.f / 12.f) * x5;
					w[dim][4] = -(1.f / 12.f) * x1 - (1.f / 24.f) * x2 - (11.f / 8.f) * x3 + (61.f / 24.f) * x4 - (25.f / 24.f) * x5;
					w[dim][5] = (7.f / 24.f) * x3 - (0.5f) * x4 + (5.f / 24.f) * x5;
					x5 = 5.0f * x4;
					x4 = 4.0f * x3;
					x3 = 3.0f * x2;
					x2 = 2.0f * x1;
					x1 = 1.0f;
					dw[dim][0] = (1.f / 12.f) * x1 - (1.f / 24.f) * x2 - (3.f / 8.f) * x3 + (13.f / 24.f) * x4 - (5.f / 24.f) * x5;
					dw[dim][1] = -(2.f / 3.f) * x1 + (2.f / 3.f) * x2 + (13.f / 8.f) * x3 - (8.f / 3.f) * x4 + (25.f / 24.f) * x5;
					dw[dim][2] = -(5.f / 4.f) * x2 - (35.f / 12.f) * x3 + (21.f / 4.f) * x4 - (25.f / 12.f) * x5;
					dw[dim][3] = (2.f / 3.f) * x1 + (2.f / 3.f) * x2 + (11.f / 4.f) * x3 - (31.f / 6.f) * x4 + (25.f / 12.f) * x5;
					dw[dim][4] = -(1.f / 12.f) * x1 - (1.f / 24.f) * x2 - (11.f / 8.f) * x3 + (61.f / 24.f) * x4 - (25.f / 24.f) * x5;
					dw[dim][5] = (7.f / 24.f) * x3 - (0.5f) * x4 + (5.f / 24.f) * x5;
				}
				for (int dim1 = 0; dim1 < NDIM; dim1++) {
					for (J[0] = I[0]; J[0] < I[0] + NINTERP; J[0]++) {
						for (J[1] = I[1]; J[1] < I[1] + NINTERP; J[1]++) {
							for (J[2] = I[2]; J[2] < I[2] + NINTERP; J[2]++) {
								double w0 = 1.0;
								for (int dim2 = 0; dim2 < NDIM; dim2++) {
									const int i0 = J[dim2] - I[dim2];
									if (dim1 == dim2) {
										w0 *= dw[dim2][i0];
									} else {
										w0 *= w[dim2][i0];
									}
								}
								const int l = params.phi_box.index(J);
								g[dim1] -= w0 * params.phi[l] * params.Nfour;
							}
						}
					}
				}
				for (J[0] = I[0]; J[0] < I[0] + NINTERP; J[0]++) {
					for (J[1] = I[1]; J[1] < I[1] + NINTERP; J[1]++) {
						for (J[2] = I[2]; J[2] < I[2] + NINTERP; J[2]++) {
							double w0 = 1.0;
							for (int dim2 = 0; dim2 < NDIM; dim2++) {
								const int i0 = J[dim2] - I[dim2];
								w0 *= w[dim2][i0];
							}
							const int l = params.phi_box.index(J);
							phi += w0 * params.phi[l];
						}
					}
				}
			}
			for (int ni = 0; ni < NCELLS; ni++) {
				const auto& src_cell = params.source_cells[cell_index * NCELLS + ni];
				for (int sibase = src_cell.begin; sibase < src_cell.end; sibase += KICK_PME_BLOCK_SIZE) {
					int source_index = sibase + tid;
					__syncthreads();
					if (source_index < src_cell.end) {
						const int j = source_index - sibase;
						assert(j >= 0);
						assert(j < KICK_PME_BLOCK_SIZE);
						shmem.x[j] = params.x[source_index];
						shmem.y[j] = params.y[source_index];
						shmem.z[j] = params.z[source_index];
					}
					__syncthreads();
					if (sink_index < nactive) {
						const int this_size = min(src_cell.end - sibase, KICK_PME_BLOCK_SIZE);
						for (int j = 0; j < this_size; j++) {
							const fixed32& src_x = shmem.x[j];
							const fixed32& src_y = shmem.y[j];
							const fixed32& src_z = shmem.z[j];
							const float dx = distance(sink_x, src_x);
							const float dy = distance(sink_y, src_y);
							const float dz = distance(sink_z, src_z);
							const float r2 = sqr(dx, dy, dz);
							float rinv, rinv3;
							if (r2 > h2) {
								const float r = sqrtf(r2);
								rinv = rsqrtf(r2);
								const float r0 = r * inv2rs;
								float exp0;
								const float erfc0 = erfcexp(r0, &exp0);
								rinv3 = (erfc0 + twooversqrtpi * r0 * exp0) * rinv * rinv * rinv;
								rinv *= erfc0;
							} else {
								const float q = sqrtf(r2) * hinv;
								const float q2 = q * q;
								rinv3 = +15.0f / 8.0f;
								rinv3 = fmaf(rinv3, q2, -21.0f / 4.0f);
								rinv3 = fmaf(rinv3, q2, +35.0f / 8.0f);
								rinv3 *= h3inv;
								rinv = -5.0f / 16.0f;
								rinv = fmaf(rinv, q2, 21.0f / 16.0f);
								rinv = fmaf(rinv, q2, -35.0f / 16.0f);
								rinv = fmaf(rinv, q2, 35.0f / 16.0f);
								rinv *= hinv;
							}
							g[XDIM] -= dx * rinv3;
							g[YDIM] -= dy * rinv3;
							g[ZDIM] -= dz * rinv3;
							phi -= rinv;
						}
					}
				}
			}
			if (sink_index < nactive) {
				g[XDIM] *= params.GM;
				g[YDIM] *= params.GM;
				g[ZDIM] *= params.GM;
				phi *= params.GM;
#ifdef FORCE_TEST
				params.gx[snki] = g[XDIM];
				params.gy[snki] = g[YDIM];
				params.gz[snki] = g[ZDIM];
				params.pot[snki] = phi;
#endif
				auto& vx = params.velx[snki];
				auto& vy = params.vely[snki];
				auto& vz = params.velz[snki];
				auto& rung = params.rung[snki];
				auto dt = 0.5f * rung_dt[rung] * params.t0;
				if (!params.first_call) {
					vx = fmaf(g[XDIM], dt, vx);
					vy = fmaf(g[YDIM], dt, vy);
					vz = fmaf(g[ZDIM], dt, vz);
				}
				const auto g2 = sqr(g[0], g[1], g[2]);
				const auto factor = params.eta * sqrtf(params.scale * params.hsoft);
				dt = fminf(factor * rsqrt(sqrtf(g2)), params.t0);
				rung = fmaxf(ceilf(log2f(params.t0) - log2f(dt)), rung - 1);
				if (rung < 0 || rung >= MAX_RUNG) {
					PRINT("Rung out of range %i\n", rung);
				}
				assert(rung >= 0);
				assert(rung < MAX_RUNG);
				dt = 0.5f * rung_dt[rung] * params.t0;
				vx = fmaf(g[XDIM], dt, vx);
				vy = fmaf(g[YDIM], dt, vy);
				vz = fmaf(g[ZDIM], dt, vz);
				//		PRINT( "%i\n", snki);
			}
		}
	}
}

struct cpymem {
	void* dest;
	void* src;
	size_t size;
};

#define NSTREAMS 16

static void process_copies(vector<cpymem> copies, cudaMemcpyKind direction, cudaStream_t stream) {
	vector<cpymem> compressed;
	std::sort(copies.begin(), copies.end(), [](cpymem a, cpymem b) {
		return a.dest < b.dest;
	});
	for (int i = 0; i < copies.size(); i++) {
		cpymem copy = copies[i];
		for (int j = i + 1; j < copies.size(); j++) {
			if (((char*) copy.dest + copy.size == copies[j].dest) && ((char*) copy.src + copy.size == copies[j].src)) {
				copy.size += copies[j].size;
				i++;
			} else {
				break;
			}
		}
		compressed.push_back(copy);
	}
	PRINT("Compressed from %i to %i copies\n", copies.size(), compressed.size());
	for (int i = 0; i < compressed.size(); i++) {
		CUDA_CHECK(cudaMemcpyAsync(compressed[i].dest, compressed[i].src, compressed[i].size, direction, stream));
	}
}

void kick_pme(range<int> box, int min_rung, double scale, double t0, bool first_call) {
	PRINT("Sorting cells\n");
	timer tm;
	size_t nsources = 0;
	size_t nsinks = 0;
	array<int, NDIM> i;
	const auto bigbox = box.pad(1);
	const size_t bigvol = bigbox.volume();
	const size_t vol = box.volume();
	print("%i\n", bigvol);
	for (i[0] = bigbox.begin[0]; i[0] != bigbox.end[0]; i[0]++) {
		for (i[1] = bigbox.begin[1]; i[1] != bigbox.end[1]; i[1]++) {
			for (i[2] = bigbox.begin[2]; i[2] != bigbox.end[2]; i[2]++) {
				auto this_cell = chainmesh_get(i);
				nsources += this_cell.pend - this_cell.pbegin;
			}
		}
	}
	for (i[0] = box.begin[0]; i[0] != box.end[0]; i[0]++) {
		for (i[1] = box.begin[1]; i[1] != box.end[1]; i[1]++) {
			for (i[2] = box.begin[2]; i[2] != box.end[2]; i[2]++) {
				auto this_cell = chainmesh_get(i);
				nsinks += this_cell.pend - this_cell.pbegin;
			}
		}
	}
	auto phibox = box;
	for (int dim = 0; dim < NDIM; dim++) {
		phibox.begin[dim] *= get_options().four_o_chain;
		phibox.end[dim] *= get_options().four_o_chain;
	}
	phibox = phibox.pad(PHI_BW);
	const size_t mem_required = mem_requirements(nsources, nsinks, vol, bigvol, phibox.volume());
	const size_t free_mem = (size_t) 85 * cuda_free_mem() / size_t(100);
	PRINT("required = %li freemem = %li\n", mem_required, free_mem);
	if (mem_required > free_mem) {
		const auto child_boxes = box.split();
		PRINT("Splitting\n");
		kick_pme(child_boxes.first, min_rung, scale, t0, first_call);
		kick_pme(child_boxes.second, min_rung, scale, t0, first_call);
	} else {
		cuda_set_device();
		PRINT("Data transfer\n");
		tm.start();
		kernel_params params;
		params.allocate(nsources, nsinks, vol, bigvol, phibox.volume());
		tm.stop();
		PRINT("%e\n", tm.read());
		tm.start();
		params.min_rung = min_rung;
		params.rs = get_options().rs;
		params.GM = get_options().GM;
		params.Nfour = get_options().four_dim;
		params.phi_box = phibox;
		params.eta = get_options().eta;
		params.first_call = first_call;
		params.t0 = t0;
		params.scale = scale;
		params.hsoft = get_options().hsoft;
		params.inv2rs = 1.0f / params.rs / 2.0f;
		params.twooversqrtpi = 2.0f / sqrtf(M_PI);
		params.h2 = sqr(params.hsoft);
		params.hinv = 1.f / params.hsoft;
		params.h3inv = params.hinv * sqr(params.hinv);
		vector<source_cell> source_cells(bigvol);
		vector<source_cell> dev_source_cells(NCELLS * vol);
		vector<sink_cell> sink_cells(vol);
		cudaStream_t stream;
		CUDA_CHECK(cudaStreamCreate(&stream));
		auto phi = gravity_long_get_phi(phibox);
		CUDA_CHECK(cudaMemcpyAsync(params.phi, phi.data(), sizeof(float) * phi.size(), cudaMemcpyHostToDevice, stream));
		size_t count = 0;
		vector<cpymem> copies;
		for (i[0] = bigbox.begin[0]; i[0] != bigbox.end[0]; i[0]++) {
			for (i[1] = bigbox.begin[1]; i[1] != bigbox.end[1]; i[1]++) {
				for (i[2] = bigbox.begin[2]; i[2] != bigbox.end[2]; i[2]++) {
					auto this_cell = chainmesh_get(i);
					const auto this_size = this_cell.pend - this_cell.pbegin;
					const auto begin = this_cell.pbegin;
					cpymem cpy;
					cpy.size = sizeof(fixed32) * this_size;
					cpy.dest = params.x + count;
					cpy.src = &particles_pos(XDIM, begin);
					copies.push_back(cpy);
					cpy.dest = params.y + count;
					cpy.src = &particles_pos(YDIM, begin);
					copies.push_back(cpy);
					cpy.dest = params.z + count;
					cpy.src = &particles_pos(ZDIM, begin);
					copies.push_back(cpy);
					const int l = bigbox.index(i);
					source_cells[l].begin = count;
					count += this_size;
					source_cells[l].end = count;
				}
			}
		}
		count = 0;
		for (i[0] = box.begin[0]; i[0] != box.end[0]; i[0]++) {
			for (i[1] = box.begin[1]; i[1] != box.end[1]; i[1]++) {
				for (i[2] = box.begin[2]; i[2] != box.end[2]; i[2]++) {
					auto this_cell = chainmesh_get(i);
					const auto this_size = this_cell.pend - this_cell.pbegin;
					const auto begin = this_cell.pbegin;
					cpymem cpy;
					cpy.size = sizeof(float) * this_size;
					cpy.dest = params.velx + count;
					cpy.src = &particles_vel(XDIM, begin);
					copies.push_back(cpy);
					cpy.dest = params.vely + count;
					cpy.src = &particles_vel(YDIM, begin);
					copies.push_back(cpy);
					cpy.dest = params.velz + count;
					cpy.src = &particles_vel(ZDIM, begin);
					copies.push_back(cpy);
					cpy.size = sizeof(char) * this_size;
					cpy.dest = params.rung + count;
					cpy.src = &particles_rung(begin);
					copies.push_back(cpy);
#ifdef FORCE_TEST
					cpy.size = sizeof(float) * this_size;
					cpy.dest = params.gx + count;
					cpy.src = &particles_gforce(XDIM, begin);
					copies.push_back(cpy);
					cpy.dest = params.gy + count;
					cpy.src = &particles_gforce(YDIM, begin);
					copies.push_back(cpy);
					cpy.dest = params.gz + count;
					cpy.src = &particles_gforce(ZDIM, begin);
					copies.push_back(cpy);
					cpy.dest = params.pot + count;
					cpy.src = &particles_pot(begin);
					copies.push_back(cpy);
#endif
					const int l = box.index(i);
					sink_cells[l].begin = count;
					count += this_size;
					sink_cells[l].end = count;
					sink_cells[l].loc = i;
					array<int, NDIM> j;
					int p = 0;
					for (j[0] = i[0] - 1; j[0] <= i[0] + 1; j[0]++) {
						for (j[1] = i[1] - 1; j[1] <= i[1] + 1; j[1]++) {
							for (j[2] = i[2] - 1; j[2] <= i[2] + 1; j[2]++) {
								const int k = bigbox.index(j);
								dev_source_cells[p + NCELLS * l] = source_cells[k];
								p++;
							}
						}
					}

				}
			}
		}
		PRINT("sink count = %i\n", count);
		CUDA_CHECK(cudaMemcpyAsync(params.sink_cells, sink_cells.data(), sizeof(sink_cell) * sink_cells.size(), cudaMemcpyHostToDevice, stream));
		CUDA_CHECK(cudaMemcpyAsync(params.source_cells, dev_source_cells.data(), sizeof(source_cell) * dev_source_cells.size(), cudaMemcpyHostToDevice, stream));
		process_copies(std::move(copies), cudaMemcpyHostToDevice, stream);
		cudaFuncAttributes attr;
		cudaFuncGetAttributes(&attr, kick_pme_kernel);
		if (attr.maxThreadsPerBlock < KICK_PME_BLOCK_SIZE) {
			PRINT("This CUDA device will not run kick_pme_kernel with the required number of threads (%i)\n", KICK_PME_BLOCK_SIZE);
			abort();
		}
		int occupancy;
		CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor ( &occupancy, kick_pme_kernel,KICK_PME_BLOCK_SIZE, sizeof(shmem_type)));
		int num_blocks = occupancy * cuda_smp_count();
		CUDA_CHECK(cudaStreamSynchronize(stream));
		tm.stop();
		PRINT("%e\n", tm.read());
		tm.reset();
		tm.start();
		PRINT("Launching kernel\n");
		CUDA_CHECK(cudaMemcpyToSymbol(dev_params, &params, sizeof(kernel_params)));
		kick_pme_kernel<<<num_blocks,KICK_PME_BLOCK_SIZE,0,stream>>>();
		count = 0;
		CUDA_CHECK(cudaStreamSynchronize(stream));
		tm.stop();
		PRINT("%e\n", tm.read());
		tm.reset();
		tm.start();
		PRINT("Transfer back\n");
		copies.resize(0);
		count = 0;
		for (i[0] = box.begin[0]; i[0] != box.end[0]; i[0]++) {
			for (i[1] = box.begin[1]; i[1] != box.end[1]; i[1]++) {
				for (i[2] = box.begin[2]; i[2] != box.end[2]; i[2]++) {
					auto this_cell = chainmesh_get(i);
					const auto this_size = this_cell.pend - this_cell.pbegin;
					const auto begin = this_cell.pbegin;
					cpymem cpy;
					cpy.size = sizeof(float) * this_size;
					cpy.src = params.velx + count;
					cpy.dest = &particles_vel(XDIM, begin);
					copies.push_back(cpy);
					cpy.src = params.vely + count;
					cpy.dest = &particles_vel(YDIM, begin);
					copies.push_back(cpy);
					cpy.src = params.velz + count;
					cpy.dest = &particles_vel(ZDIM, begin);
					copies.push_back(cpy);
					cpy.size = sizeof(char) * this_size;
					cpy.src = params.rung + count;
					cpy.dest = &particles_rung(begin);
					copies.push_back(cpy);
#ifdef FORCE_TEST
					cpy.size = sizeof(float) * this_size;
					cpy.src = params.gx + count;
					cpy.dest = &particles_gforce(XDIM, begin);
					copies.push_back(cpy);
					cpy.src = params.gy + count;
					cpy.dest = &particles_gforce(YDIM, begin);
					copies.push_back(cpy);
					cpy.src = params.gz + count;
					cpy.dest = &particles_gforce(ZDIM, begin);
					copies.push_back(cpy);
					cpy.src = params.pot + count;
					cpy.dest = &particles_pot(begin);
					copies.push_back(cpy);
#endif
					count += this_size;
				}
			}
		}
		process_copies(std::move(copies), cudaMemcpyDeviceToHost, stream);
		CUDA_CHECK(cudaStreamSynchronize(stream));
		params.free();
		CUDA_CHECK(cudaStreamDestroy(stream));
		tm.stop();
		PRINT("%e\n", tm.read());
	}
}
