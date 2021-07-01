#include <tigerpm/kick_pme.hpp>
#include <tigerpm/range.hpp>
#include <tigerpm/chainmesh.hpp>
#include <tigerpm/particles.hpp>
#include <tigerpm/gravity_short.hpp>
#include <tigerpm/gravity_long.hpp>
#include <tigerpm/util.hpp>

#define MAX_RUNG 24
#define NINTERP 4
#define NCELLS 27
#define KICK_PME_BLOCK_SIZE 128

__constant__ float rung_dt[MAX_RUNG] = { 1.0 / (1 << 0), 1.0 / (1 << 1), 1.0 / (1 << 2), 1.0 / (1 << 3), 1.0 / (1 << 4),
		1.0 / (1 << 5), 1.0 / (1 << 6), 1.0 / (1 << 7), 1.0 / (1 << 8), 1.0 / (1 << 9), 1.0 / (1 << 10), 1.0 / (1 << 11),
		1.0 / (1 << 12), 1.0 / (1 << 13), 1.0 / (1 << 14), 1.0 / (1 << 15), 1.0 / (1 << 16), 1.0 / (1 << 17), 1.0
				/ (1 << 18), 1.0 / (1 << 19), 1.0 / (1 << 20), 1.0 / (1 << 21), 1.0 / (1 << 22), 1.0 / (1 << 23) };

struct shmem_type {
	fixed32 x[KICK_PME_BLOCK_SIZE];
	fixed32 y[KICK_PME_BLOCK_SIZE];
	fixed32 z[KICK_PME_BLOCK_SIZE];
	int index[KICK_PME_BLOCK_SIZE];
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
	bool first_call;
	int Nfour;
	range<int> bigbox;
#ifdef TEST_FORCE
	float* gx;
	float* gy;
	float* gz;
	float* pot;
#endif
	void allocate(size_t source_size, size_t sink_size, size_t cell_count, size_t big_cell_count) {
		nsink_cells = cell_count;
		CUDA_CHECK(cudaMalloc(&x, source_size * sizeof(fixed32)));
		CUDA_CHECK(cudaMalloc(&y, source_size * sizeof(fixed32)));
		CUDA_CHECK(cudaMalloc(&z, source_size * sizeof(fixed32)));
		CUDA_CHECK(cudaMalloc(&velx, sink_size * sizeof(float)));
		CUDA_CHECK(cudaMalloc(&vely, sink_size * sizeof(float)));
		CUDA_CHECK(cudaMalloc(&velz, sink_size * sizeof(float)));
		CUDA_CHECK(cudaMalloc(&rung, sink_size * sizeof(char)));
		CUDA_CHECK(cudaMalloc(&phi, big_cell_count * sizeof(float)));
		CUDA_CHECK(cudaMalloc(&active_sinki, sink_size * sizeof(char)));
		CUDA_CHECK(cudaMalloc(&active_sourcei, sink_size * sizeof(char)));
		CUDA_CHECK(cudaMalloc(&source_cells, cell_count * NCELLS * sizeof(source_cell)));
		CUDA_CHECK(cudaMalloc(&sink_cells, cell_count * sizeof(source_cell)));
#ifdef TEST_FORCE
		CUDA_CHECK(cudaMalloc(&gx,source_size*sizeof(float)));
		CUDA_CHECK(cudaMalloc(&gy,source_size*sizeof(float)));
		CUDA_CHECK(cudaMalloc(&gz,source_size*sizeof(float)));
		CUDA_CHECK(cudaMalloc(&pot,source_size*sizeof(float)));
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
#ifdef TEST_FORCE
		CUDA_CHECK(cudaFree(gx));
		CUDA_CHECK(cudaFree(gy));
		CUDA_CHECK(cudaFree(gz));
		CUDA_CHECK(cudaFree(pot));
#endif
	}
};

static size_t mem_requirements(range<int> box) {
	size_t mem = 0;
	const auto bigbox = box.pad(1);
	mem += NDIM * sizeof(fixed32) * bigbox.volume();
	mem += NDIM * sizeof(float) * box.volume();
	mem += sizeof(char) * box.volume();
	mem += NCELLS * bigbox.volume() * sizeof(source_cell);
	mem += box.volume() * sizeof(sink_cell);
	mem += 2 * sizeof(int) * box.volume();
	mem += bigbox.volume() * sizeof(float);
	mem += sizeof(kernel_params);
#ifdef TEST_FORCE
	mem += (NDIM+1) * sizeof(float) * box.volume();
#endif
	return mem;
}

__global__ void kick_pme_kernel(kernel_params params) {
	__shared__ shmem_type shmem;
	const int& tid = threadIdx.x;
	const int& bid = blockIdx.x;
	const int& gsz = gridDim.x;
	const int cell_begin = size_t(bid) * (size_t) params.nsink_cells / (size_t) gsz;
	const int cell_end = size_t(bid + 1) * (size_t) params.nsink_cells / (size_t) gsz;
	const float inv2rs = 1.0f / params.rs / 2.0f;
	const float twooversqrtpi = 2.0f / sqrtf(M_PI);
	const float h2 = sqr(params.hsoft);
	const float hinv = 1.f / params.hsoft;
	const float h3inv = hinv * sqr(hinv);
	for (int cell_index = cell_begin; cell_index < cell_end; cell_index++) {
		int nactive = 0;
		const sink_cell& sink = params.sink_cells[cell_index];
		const auto& my_source_cell = params.source_cells[cell_index * NCELLS + NCELLS / 2];
		const int nsinks = sink.end - sink.begin;
		for (int i = tid; i < nsinks; i += KICK_PME_BLOCK_SIZE) {
			const int this_index = sink.begin + i;
			const bool is_active = int(params.rung[this_index] >= params.min_rung);
			shmem.index[tid] = int(is_active);
			for (int P = 1; P < KICK_PME_BLOCK_SIZE; P *= 2) {
				int tmp;
				__syncthreads();
				if (tid >= P) {
					tmp = shmem.index[tid - 1];
				}
				__syncthreads();
				if (tid >= P) {
					shmem.index[tid] += tmp;
				}
			}
			__syncthreads();
			int this_count = shmem.index[KICK_PME_BLOCK_SIZE - 1];
			const int active_index = tid > 0 ? shmem.index[tid - 1] : 0;
			if (is_active) {
				params.active_sinki[active_index] = this_index;
				params.active_sourcei[active_index] = my_source_cell.begin + i;
			}
			nactive += this_count;
		}
		const int maxsink = round_up(nactive, KICK_PME_BLOCK_SIZE);
		for (int sink_index = tid; sink_index < maxsink; sink_index += KICK_PME_BLOCK_SIZE) {
			float g[NDIM] = { 0.f, 0.f, 0.f };
			float phi = 0.f;
			int srci;
			int snki;
			fixed32 sink_x;
			fixed32 sink_y;
			fixed32 sink_z;
			if (sink_index < nactive) {
				srci = params.active_sourcei[sink_index];
				snki = params.active_sinki[sink_index];
				sink_x = params.x[srci];
				sink_y = params.y[srci];
				sink_z = params.z[srci];
				int I[NDIM];
				int J[NDIM];
				float X[NDIM] = { sink_x.to_float(), sink_y.to_float(), sink_z.to_float() };
				float w[NINTERP][NINTERP];
				float dw[NINTERP][NINTERP];
				for (int dim = 0; dim < NDIM; dim++) {
					X[dim] *= params.Nfour;
					I[dim] = int(X[dim]);
					X[dim] -= float(I[dim]);
					I[dim]--;
				}
				for (int dim = 0; dim < NDIM; dim++) {
					const float& x1 = X[dim];
					const float x2 = X[dim] * x1;
					const float x3 = x1 * x2;
					w[dim][0] = -0.5f * x1 + x2 - 0.5f * x3;
					w[dim][1] = 1.0f - 2.5f * x2 + 1.5f * x3;
					w[dim][2] = 0.5f * x1 + 2.0f * x2 - 1.5f * x3;
					w[dim][3] = -0.5f * x2 + 0.5f * x3;
					dw[dim][0] = -0.5f + 2.0f * x1 - 1.5f * x2;
					dw[dim][1] = -5.0f * x1 + 4.5f * x2;
					dw[dim][2] = 0.5f + 4.0f * x1 - 4.5f * x2;
					dw[dim][3] = -x1 + 1.5f * x2;
				}
				for (int dim1 = 0; dim1 < NDIM; dim1++) {
					g[dim1] = 0.0;
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
								const int l = params.bigbox.index(J);
								g[dim1] -= w0 * params.phi[l] * params.Nfour;
							}
						}
					}
				}
				phi = 0.0;
				for (J[0] = I[0]; J[0] < I[0] + NINTERP; J[0]++) {
					for (J[1] = I[1]; J[1] < I[1] + NINTERP; J[1]++) {
						for (J[2] = I[2]; J[2] < I[2] + NINTERP; J[2]++) {
							double w0 = 1.0;
							for (int dim2 = 0; dim2 < NDIM; dim2++) {
								const int i0 = J[dim2] - I[dim2];
								w0 *= w[dim2][i0];
							}
							const int l = params.bigbox.index(J);
							phi += w0 * params.phi[l];
						}
					}
				}
			}
			for (int ni = 0; ni < NCELLS; ni++) {
				const auto& src_cell = params.source_cells[cell_index * NCELLS + ni];
				for (int sibase = src_cell.begin; sibase < src_cell.end; sibase += KICK_PME_BLOCK_SIZE) {
					if (sink_index < nactive) {
						int source_index = sibase + tid;
						if (tid < min(src_cell.end, sibase + KICK_PME_BLOCK_SIZE)) {
							const int j = source_index - src_cell.begin;
							shmem.x[j] = params.x[source_index];
							shmem.y[j] = params.y[source_index];
							shmem.z[j] = params.z[source_index];
						}
					}
					__syncthreads();
					if (sink_index < nactive) {
						const int this_size = src_cell.end - src_cell.begin;
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
								const float r02 = r0 * r0;
								const float erfc0 = erfcf(r0);
								const float exp0 = expf(-r02);
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
#ifdef TEST_FORCE
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
				dt = 0.5f * rung_dt[rung] * params.t0;
				vx = fmaf(g[XDIM], dt, vx);
				vy = fmaf(g[YDIM], dt, vy);
				vz = fmaf(g[ZDIM], dt, vz);
			}
		}
	}
}

void kick_pme(range<int> box, int min_rung, double scale, double t0, bool first_call) {
	const size_t mem_required = mem_requirements(box);
	if (mem_required > cuda_free_mem() * 85 / 100) {
		const auto child_boxes = box.split();
		kick_pme(child_boxes.first, min_rung, scale, t0, first_call);
		kick_pme(child_boxes.second, min_rung, scale, t0, first_call);
	} else {
		cuda_set_device();
		const auto bigbox = box.pad(1);
		const size_t bigvol = bigbox.volume();
		const size_t vol = box.volume();
		kernel_params params;
		size_t nsources = 0;
		size_t nsinks = 0;
		array<int, NDIM> i;
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
		params.allocate(nsources, nsinks, vol, bigvol);
		params.min_rung = min_rung;
		params.rs = get_options().rs;
		params.GM = get_options().GM;
		params.Nfour = get_options().four_dim;
		params.bigbox = bigbox;
		params.eta = get_options().eta;
		params.first_call = first_call;
		params.t0 = t0;
		params.scale = scale;
		params.hsoft = get_options().hsoft;
		std::vector<source_cell> source_cells(bigvol);
		std::vector<source_cell> dev_source_cells(NCELLS * vol);
		std::vector<sink_cell> sink_cells(bigvol);
		cudaStream_t stream;
		cudaStreamCreate(&stream);
		auto phi = gravity_long_get_phi(bigbox);
		CUDA_CHECK(cudaMemcpyAsync(params.phi, phi.data(), sizeof(float) * phi.size(), cudaMemcpyHostToDevice, stream));
		size_t count = 0;
		for (i[0] = bigbox.begin[0]; i[0] != bigbox.end[0]; i[0]++) {
			for (i[1] = bigbox.begin[1]; i[1] != bigbox.end[1]; i[1]++) {
				for (i[2] = bigbox.begin[2]; i[2] != bigbox.end[2]; i[2]++) {
					auto this_cell = chainmesh_get(i);
					const auto this_size = this_cell.pend - this_cell.pbegin;
					const auto begin = this_cell.pbegin;
					CUDA_CHECK(
							cudaMemcpyAsync(params.x + count, &particles_pos(XDIM, begin), sizeof(fixed32) * this_size, cudaMemcpyHostToDevice, stream));
					CUDA_CHECK(
							cudaMemcpyAsync(params.y + count, &particles_pos(YDIM, begin), sizeof(fixed32) * this_size, cudaMemcpyHostToDevice, stream));
					CUDA_CHECK(
							cudaMemcpyAsync(params.z + count, &particles_pos(ZDIM, begin), sizeof(fixed32) * this_size, cudaMemcpyHostToDevice, stream));
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
					CUDA_CHECK(
							cudaMemcpyAsync(params.velx + count, &particles_vel(XDIM, begin), sizeof(float) * this_size, cudaMemcpyHostToDevice, stream));
					CUDA_CHECK(
							cudaMemcpyAsync(params.vely + count, &particles_vel(YDIM, begin), sizeof(float) * this_size, cudaMemcpyHostToDevice, stream));
					CUDA_CHECK(
							cudaMemcpyAsync(params.velz + count, &particles_vel(ZDIM, begin), sizeof(float) * this_size, cudaMemcpyHostToDevice, stream));
					CUDA_CHECK(
							cudaMemcpyAsync(params.rung + count, &particles_rung(begin), sizeof(char) * this_size,
									cudaMemcpyHostToDevice, stream));
#ifdef TEST_FORCE
					CUDA_CHECK(
							cudaMemcpyAsync(params.gx + count, &particles_gforce(XDIM, begin), sizeof(float) * this_size, cudaMemcpyHostToDevice, stream));
					CUDA_CHECK(
							cudaMemcpyAsync(params.gy + count, &particles_gforce(YDIM, begin), sizeof(float) * this_size, cudaMemcpyHostToDevice, stream));
					CUDA_CHECK(
							cudaMemcpyAsync(params.gz + count, &particles_gforce(ZDIM, begin), sizeof(float) * this_size, cudaMemcpyHostToDevice, stream));
					CUDA_CHECK(
							cudaMemcpyAsync(params.pot + count, &particles_pot(begin), sizeof(float) * this_size, cudaMemcpyHostToDevice, stream));
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
							for (j[2] = i[1] - 1; j[1] <= i[2] + 1; j[2]++) {
								const int k = bigbox.index(j);
								dev_source_cells[p + NCELLS * l] = source_cells[k];
								p++;
							}
						}
					}

				}
			}
		}
		CUDA_CHECK(
				cudaMemcpyAsync(params.sink_cells, sink_cells.data(), sizeof(sink_cell) * sink_cells.size(),
						cudaMemcpyHostToDevice));
		CUDA_CHECK(
				cudaMemcpyAsync(params.source_cells, dev_source_cells.data(), sizeof(source_cell) * dev_source_cells.size(),
						cudaMemcpyHostToDevice));
		int occupancy;
		CUDA_CHECK(
				cudaOccupancyMaxActiveBlocksPerMultiprocessor ( &occupancy, kick_pme_kernel,KICK_PME_BLOCK_SIZE, sizeof(shmem_type)));
		int num_blocks = occupancy * cuda_smp_count();
		kick_pme_kernel<<<num_blocks,KICK_PME_BLOCK_SIZE,0,stream>>>(params);
		count = 0;
		for (i[0] = box.begin[0]; i[0] != box.end[0]; i[0]++) {
			for (i[1] = box.begin[1]; i[1] != box.end[1]; i[1]++) {
				for (i[2] = box.begin[2]; i[2] != box.end[2]; i[2]++) {
					auto this_cell = chainmesh_get(i);
					const auto this_size = this_cell.pend - this_cell.pbegin;
					const auto begin = this_cell.pbegin;
					CUDA_CHECK(
							cudaMemcpyAsync(&particles_vel(XDIM, begin), params.velx + count,sizeof(float) * this_size, cudaMemcpyDeviceToHost, stream));
					CUDA_CHECK(
							cudaMemcpyAsync(&particles_vel(YDIM, begin), params.vely + count, sizeof(float) * this_size, cudaMemcpyDeviceToHost, stream));
					CUDA_CHECK(
							cudaMemcpyAsync(&particles_vel(ZDIM, begin), params.velz + count, sizeof(float) * this_size, cudaMemcpyDeviceToHost, stream));
					CUDA_CHECK(
							cudaMemcpyAsync(&particles_rung(begin), params.rung + count, sizeof(char) * this_size,
									cudaMemcpyDeviceToHost));
#ifdef TEST_FORCE
					CUDA_CHECK(
							cudaMemcpyAsync(&particles_gforce(XDIM, begin), params.gx + count, sizeof(float) * this_size, cudaMemcpyDeviceToHost, stream));
					CUDA_CHECK(
							cudaMemcpyAsync(&particles_gforce(YDIM, begin), params.gy + count, sizeof(float) * this_size, cudaMemcpyDeviceToHost, stream));
					CUDA_CHECK(
							cudaMemcpyAsync(&particles_gforce(ZDIM, begin), params.gz + count, sizeof(float) * this_size, cudaMemcpyDeviceToHost, stream));
					CUDA_CHECK(
							cudaMemcpyAsync(&particles_pot(begin), params.pot + count, sizeof(float) * this_size, cudaMemcpyDeviceToHost, stream));
#endif
				}
			}
		}
		cudaStreamSynchronize(stream);
		params.free();
		cudaStreamDestroy(stream);
	}
}
