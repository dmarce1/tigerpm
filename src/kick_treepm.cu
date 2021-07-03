#include <tigerpm/kick_treepm.hpp>
#include <tigerpm/timer.hpp>
#include <tigerpm/options.hpp>
#include <tigerpm/particles.hpp>
#include <tigerpm/util.hpp>
#include <tigerpm/gravity_long.hpp>

#include <thrust/device_vector.h>

#include <algorithm>

#define TREEPM_BLOCK_SIZE 32

#define NCELLS 27

struct source_cell {
	int begin;
	int end;
};

struct sink_cell {
	int begin;
	int end;
	array<int, NDIM> loc;
};

struct treepm_params {
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
	sink_bucket** buckets;
	tree* trees;
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
	float theta;
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
		CUDA_CHECK(cudaMalloc(&buckets, cell_count * sizeof(sink_bucket*)));
		CUDA_CHECK(cudaMalloc(&trees, cell_count * sizeof(tree)));
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
		CUDA_CHECK(cudaFree(trees));
		CUDA_CHECK(cudaFree(buckets));
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
	mem += sizeof(sink_bucket*) * vol;
	mem += sizeof(tree) * vol;
	mem += phivol * sizeof(float);
	mem += sizeof(treepm_params);
#ifdef FORCE_TEST
	mem += (NDIM + 1) * sizeof(float) * nsinks;
#endif
	return mem;
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

void kick_treepm(vector<tree>& trees, vector<vector<sink_bucket>>& buckets, range<int> box, int min_rung, double scale, double t0, bool first_call) {
	PRINT("Sorting cells\n");
	timer tm;
	size_t nsources = 0;
	size_t nsinks = 0;
	array<int, NDIM> i;
	const auto bigbox = box.pad(1);
	const size_t bigvol = bigbox.volume();
	const size_t vol = box.volume();
	int tree_size = 0;
	int buckets_size = 0;
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
				const int index = box.index(i);
				tree_size += trees[index].size();
				buckets_size += sizeof(sink_bucket) * buckets[index].size();
			}
		}
	}
	auto phibox = box;
	for (int dim = 0; dim < NDIM; dim++) {
		phibox.begin[dim] *= get_options().four_o_chain;
		phibox.end[dim] *= get_options().four_o_chain;
	}
	phibox = phibox.pad(PHI_BW);
	const size_t mem_required = mem_requirements(nsources, nsinks, vol, bigvol, phibox.volume()) + tree_size + buckets_size;
	const size_t free_mem = (size_t) 85 * cuda_free_mem() / size_t(100);
	PRINT("required = %li freemem = %li\n", mem_required, free_mem);
	if (mem_required > free_mem) {
		const auto child_boxes = box.split();
		PRINT("Splitting\n");
		kick_treepm(trees, buckets, child_boxes.first, min_rung, scale, t0, first_call);
		kick_treepm(trees, buckets, child_boxes.second, min_rung, scale, t0, first_call);
	} else {
		cuda_set_device();
		PRINT("Data transfer\n");
		tm.start();
		treepm_params params;
		params.allocate(nsources, nsinks, vol, bigvol, phibox.volume());
		tm.stop();
		PRINT("%e\n", tm.read());
		tm.start();
		params.theta = 0.5;
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
					const int l = box.index(i);
					const auto dif = count - begin;
					trees[l].adjust_indexes(dif);
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
		vector<tree> dev_trees(vol);
		vector<thrust::device_vector<sink_bucket>> dev_buckets;
		vector<sink_bucket*> dev_bucket_ptrs;
		for (int j = 0; j < vol; j++) {
			dev_trees[j] = trees[j].to_device(stream);
			thrust::device_vector < sink_bucket > dev_bucket(std::move(buckets[j]));
			dev_buckets.push_back(dev_bucket);
			sink_bucket* ptr =  thrust::raw_pointer_cast(dev_buckets.back().data());
			dev_bucket_ptrs.push_back(ptr);
		}
		CUDA_CHECK(cudaMemcpyAsync(params.buckets, dev_bucket_ptrs.data(), sizeof(sink_bucket*) * vol, cudaMemcpyHostToDevice, stream));
		CUDA_CHECK(cudaMemcpyAsync(params.trees, dev_trees.data(), sizeof(tree) * vol, cudaMemcpyHostToDevice, stream));
		PRINT("sink count = %i\n", count);
		CUDA_CHECK(cudaMemcpyAsync(params.sink_cells, sink_cells.data(), sizeof(sink_cell) * sink_cells.size(), cudaMemcpyHostToDevice, stream));
		CUDA_CHECK(cudaMemcpyAsync(params.source_cells, dev_source_cells.data(), sizeof(source_cell) * dev_source_cells.size(), cudaMemcpyHostToDevice, stream));
		process_copies(std::move(copies), cudaMemcpyHostToDevice, stream);

		/*cudaFuncAttributes attr;
		 cudaFuncGetAttributes(&attr, kick_pme_kernel);
		 if (attr.maxThreadsPerBlock < TREEPM_BLOCK_SIZE) {
		 PRINT("This CUDA device will not run kick_pme_kernel with the required number of threads (%i)\n", TREEPM_BLOCK_SIZE);
		 abort();
		 }
		 int occupancy;
		 CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor ( &occupancy, kick_pme_kernel,TREEPM_BLOCK_SIZE, sizeof(shmem_type)));
		 int num_blocks = occupancy * cuda_smp_count();
		 CUDA_CHECK(cudaStreamSynchronize(stream));
		 tm.stop();
		 PRINT("%e\n", tm.read());
		 tm.reset();
		 tm.start();
		 PRINT("Launching kernel\n");
		 CUDA_CHECK(cudaMemcpyToSymbol(dev_params, &params, sizeof(treepm_params)));
		 kick_pme_kernel<<<num_blocks,TREEPM_BLOCK_SIZE,0,stream>>>();*/

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
