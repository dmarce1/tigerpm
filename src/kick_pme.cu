#include <tigerpm/kick_pme.hpp>
#include <tigerpm/range.hpp>
#include <tigerpm/chainmesh.hpp>
#include <tigerpm/particles.hpp>
#include <tigerpm/gravity_short.hpp>
#include <tigerpm/util.hpp>

#define NCELLS 27
#define KICK_PME_BLOCK_SIZE 96

struct shmem_type {
	fixed32 x[KICK_PME_BLOCK_SIZE];
	fixed32 y[KICK_PME_BLOCK_SIZE];
	fixed32 z[KICK_PME_BLOCK_SIZE];
};

struct source_cell {
	int begin;
	int end;
};

struct sink_cell {
	int begin;
	int end;
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
#ifdef TEST_FORCE
	mem += (NDIM+1) * sizeof(float) * box.volume();
#endif
	return mem;
}

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

__global__ void kick_pme_kernel(kernel_params params) {
	__shared__ shmem_type shmem;
	const int& tid = threadIdx.x;
	const int& bid = blockIdx.x;
	const int& gsz = gridDim.x;
	const int cell_begin = size_t(bid) * (size_t) params.nsink_cells / (size_t) gsz;
	const int cell_end = size_t(bid + 1) * (size_t) params.nsink_cells / (size_t) gsz;
	const float inv2rs = 1.0f / params.rs / 2.0f;
	const float twooversqrtpi = 2.0f / sqrtf(M_PI);
	for (int cell_index = cell_begin; cell_index < cell_end; cell_index++) {
		int nactive = 0;
		const sink_cell& sink = params.sink_cells[cell_index];
		const auto& my_source_cell = params.source_cells[cell_index * NCELLS + NCELLS / 2];
		const int nsinks = sink.end - sink.begin;
		for (int i = tid; i < nsinks; i += KICK_PME_BLOCK_SIZE) {
			const int this_index = sink.begin + i;
			const bool is_active = int(params.rung[this_index] >= params.min_rung);
			int active_index = int(is_active);
			for (int P = 1; P < KICK_PME_BLOCK_SIZE; P *= 2) {
				const auto tmp = __shfl_up_sync(0xFFFFFFFF, active_index, P);
				if (tid >= P) {
					active_index += tmp;
				}
			}
			int this_count = __shfl_sync(0xFFFFFFFF, active_index, KICK_PME_BLOCK_SIZE - 1);
			const auto tmp = __shfl_up_sync(0xFFFFFFFF, active_index, 1);
			active_index = tid > 0 ? tmp : 0;
			active_index += nactive;
			if (is_active) {
				params.active_sinki[active_index] = this_index;
				params.active_sourcei[active_index] = my_source_cell.begin + i;
			}
			nactive += this_count;
		}
		for (int sink_index = tid; sink_index < nactive; sink_index += KICK_PME_BLOCK_SIZE) {
			float fx = 0.f;
			float fy = 0.f;
			float fz = 0.f;
			float phi = 0.f;
			const int& srci = params.active_sourcei[sink_index];
			const int& snki = params.active_sinki[sink_index];
			const fixed32& sink_x = params.x[srci];
			const fixed32& sink_y = params.y[srci];
			const fixed32& sink_z = params.z[srci];
			for (int ni = 0; ni < NCELLS; ni++) {
				const auto& source_cell = params.source_cells[cell_index * NCELLS + ni];
				for (int source_index = source_cell.begin + tid; source_index < source_cell.end; source_index +=
				KICK_PME_BLOCK_SIZE) {
					const int j = source_index - source_cell.begin;
					shmem.x[j] = params.x[source_index];
					shmem.y[j] = params.y[source_index];
					shmem.z[j] = params.z[source_index];
				}
				__syncthreads();
				const int this_size = source_cell.end - source_cell.begin;
				for (int j = 0; j < this_size; j++) {
					const fixed32& src_x = shmem.x[j];
					const fixed32& src_y = shmem.y[j];
					const fixed32& src_z = shmem.z[j];
					const float dx = distance(sink_x, src_x);
					const float dy = distance(sink_y, src_y);
					const float dz = distance(sink_z, src_z);
					const float r2 = sqr(dx, dy, dz);
					const float r = sqrtf(r2);
					float rinv = rsqrtf(r2);
					const float r0 = r * inv2rs;
					const float r02 = r0 * r0;
					const float erfc0 = erfcf(r0);
					const float exp0 = expf(-r02);
					const float rinv3 = (erfc0 + twooversqrtpi * r0 * exp0) * rinv * rinv * rinv;
					rinv *= erfc0;
					fx -= dx * rinv3;
					fy -= dy * rinv3;
					fz -= dz * rinv3;
					phi -= rinv;
				}
			}
			fx *= params.GM;
			fy *= params.GM;
			fz *= params.GM;
			phi *= params.GM;

			/* DO KICK */


#ifdef TEST_FORCE
			params.gx[snki] = fx;
			params.gy[snki] = fy;
			params.gz[snki] = fz;
			params.pot[snki] = pot;
#endif
		}
	}
}

void kick_pme(range<int> box, int min_rung, float rs, float GM) {
	const size_t mem_required = mem_requirements(box);
	if (mem_required > cuda_free_mem() * 85 / 100) {
		const auto child_boxes = box.split();
		kick_pme(child_boxes.first, min_rung, rs, GM);
		kick_pme(child_boxes.second, min_rung, rs, GM);
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
		params.rs = rs;
		params.GM = GM;
		std::vector<source_cell> source_cells(bigvol);
		std::vector<source_cell> dev_source_cells(NCELLS * vol);
		std::vector<sink_cell> sink_cells(bigvol);
		size_t count = 0;
		cudaStream_t stream;
		cudaStreamCreate(&stream);
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
