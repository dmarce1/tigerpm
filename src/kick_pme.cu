#include <tigerpm/kick_pme.hpp>
#include <tigerpm/range.hpp>
#include <tigerpm/chainmesh.hpp>
#include <tigerpm/particles.hpp>

#define NCELLS 27

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
#ifdef TEST_FORCE
	mem += (NDIM+1) * sizeof(float) * box.volume();
#endif
	return mem;
}

struct kick_pme_kernel_params {
	fixed32* x;
	fixed32* y;
	fixed32* z;
	float* velx;
	float* vely;
	float* velz;
	char* rung;
	source_cell* source_cells;
	sink_cell* sink_cells;
#ifdef TEST_FORCE
	float* gx;
	float* gy;
	float* gz;
	float* pot;
#endif
	void allocate(size_t source_size, size_t sink_size, size_t cell_count) {
		CUDA_CHECK(cudaMalloc(&x, source_size * sizeof(fixed32)));
		CUDA_CHECK(cudaMalloc(&y, source_size * sizeof(fixed32)));
		CUDA_CHECK(cudaMalloc(&z, source_size * sizeof(fixed32)));
		CUDA_CHECK(cudaMalloc(&velx, source_size * sizeof(float)));
		CUDA_CHECK(cudaMalloc(&vely, source_size * sizeof(float)));
		CUDA_CHECK(cudaMalloc(&velz, source_size * sizeof(float)));
		CUDA_CHECK(cudaMalloc(&rung, source_size * sizeof(char)));
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

void kick_pme(range<int> box) {
	const size_t mem_required = mem_requirements(box);
	if (mem_required > cuda_free_mem() * 85 / 100) {
		const auto child_boxes = box.split();
		kick_pme(child_boxes.first);
		kick_pme(child_boxes.second);
	} else {
		cuda_set_device();
		const auto bigbox = box.pad(1);
		const size_t bigvol = bigbox.volume();
		const size_t vol = box.volume();
		kick_pme_kernel_params params;
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
		params.allocate(nsources, nsinks, vol);
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

		params.free();
		cudaStreamDestroy(stream);
	}
}
