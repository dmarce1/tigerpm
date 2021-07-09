#include <tigerpm/fmmpm.hpp>
#include <tigerpm/timer.hpp>
#include <tigerpm/options.hpp>
#include <tigerpm/particles.hpp>
#include <tigerpm/util.hpp>
#include <tigerpm/gravity_long.hpp>
#include <tigerpm/gravity_short.hpp>
#include <tigerpm/stack_vector.hpp>

#include <thrust/device_vector.h>

#define FMMPM_MIN_THREADS 16
#define FMMPM_BLOCK_SIZE 32

#include <algorithm>

__constant__ float rung_dt[MAX_RUNG] = { 1.0 / (1 << 0), 1.0 / (1 << 1), 1.0 / (1 << 2), 1.0 / (1 << 3), 1.0 / (1 << 4), 1.0 / (1 << 5), 1.0 / (1 << 6), 1.0
		/ (1 << 7), 1.0 / (1 << 8), 1.0 / (1 << 9), 1.0 / (1 << 10), 1.0 / (1 << 11), 1.0 / (1 << 12), 1.0 / (1 << 13), 1.0 / (1 << 14), 1.0 / (1 << 15), 1.0
		/ (1 << 16), 1.0 / (1 << 17), 1.0 / (1 << 18), 1.0 / (1 << 19), 1.0 / (1 << 20), 1.0 / (1 << 21), 1.0 / (1 << 22), 1.0 / (1 << 23), 1.0 / (1 << 24), 1.0
		/ (1 << 25), 1.0 / (1 << 26), 1.0 / (1 << 27), 1.0 / (1 << 28), 1.0 / (1 << 29), 1.0 / (1 << 30), 1.0 / (1 << 31) };

#define LIST_SIZE  (4*1024)
#define STACK_SIZE (32*1024)
#define MAX_DEPTH 64

struct checkitem {
	int index :24;
	int opened :8;
	tree* tr;

	CUDA_EXPORT inline
	bool is_leaf() const {
		return tr->is_leaf(index);
	}
	CUDA_EXPORT inline
	fixed32 get_x(int dim) const {
		return tr->get_x(dim, index);
	}
	CUDA_EXPORT inline
	float get_radius() const {
		return tr->get_radius(index);
	}
	CUDA_EXPORT inline
	array<checkitem, 2> get_children() {
		const auto indices = tr->get_children(index);
		array<checkitem, 2> c;
		c[0].index = indices[0];
		c[1].index = indices[1];
		c[0].opened = c[1].opened = 0;
	}
};

struct list_set {
	stack_vector<checkitem, STACK_SIZE, MAX_DEPTH> checklist;
	fixedcapvec<checkitem, LIST_SIZE> openlist;
	fixedcapvec<checkitem, LIST_SIZE> nextlist;
	fixedcapvec<checkitem, LIST_SIZE> multilist;
	fixedcapvec<checkitem, LIST_SIZE> partlist;
};

struct fmmpm_params {
	fixed32* x;
	fixed32* y;
	fixed32* z;
	float* velx;
	float* vely;
	float* velz;
	float* phi;
	char* rung;
	list_set* lists;
	tree* tree_neighbors;
	int* active;
	int min_rung;
	bool do_phi;
	float rs;
	float rcut;
	float GM;
	float eta;
	float t0;
	float scale;
	float hsoft;
	float inv2rs;
	float phi0;
	float twooversqrtpi;
	float h2;
	float hinv;
	float h2inv;
	float h3inv;
	float theta;
	int nsink_cells;
	bool first_call;
	int Nfour;
	range<int> phi_box;
#ifdef FORCE_TEST
	float* gx;
	float* gy;
	float* gz;
	float* pot;
#endif
	void allocate(size_t source_size, size_t sink_size, size_t cell_count, size_t big_cell_count, size_t phi_cell_count, int nblocks) {
		CUDA_CHECK(cudaMalloc(&tree_neighbors, cell_count * NCELLS * sizeof(tree)));
		nsink_cells = cell_count;
		CUDA_CHECK(cudaMalloc(&x, source_size * sizeof(fixed32)));
		CUDA_CHECK(cudaMalloc(&y, source_size * sizeof(fixed32)));
		CUDA_CHECK(cudaMalloc(&z, source_size * sizeof(fixed32)));
		CUDA_CHECK(cudaMalloc(&velx, sink_size * sizeof(float)));
		CUDA_CHECK(cudaMalloc(&vely, sink_size * sizeof(float)));
		CUDA_CHECK(cudaMalloc(&velz, sink_size * sizeof(float)));
		CUDA_CHECK(cudaMalloc(&lists, sizeof(list_set)));
		CUDA_CHECK(cudaMalloc(&rung, sink_size * sizeof(char)));
		CUDA_CHECK(cudaMalloc(&phi, phi_cell_count * sizeof(float)));
		CUDA_CHECK(cudaMalloc(&active, nblocks * sizeof(int) * SINK_BUCKET_SIZE));
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
		CUDA_CHECK(cudaFree(active));
		CUDA_CHECK(cudaFree(rung));
		CUDA_CHECK(cudaFree(lists));
		CUDA_CHECK(cudaFree(tree_neighbors));
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
	mem += NCELLS * bigvol * sizeof(tree);
	mem += 2 * sizeof(int) * vol;
	mem += sizeof(int) * vol;
	mem += sizeof(tree) * vol;
	mem += phivol * sizeof(float);
	mem += sizeof(fmmpm_params);
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

struct fmmpm_shmem {
	array<array<float, NDIM>, SINK_BUCKET_SIZE> g;
	array<float, SINK_BUCKET_SIZE> phi;
	array<fixed32, SINK_BUCKET_SIZE> x;
	array<fixed32, SINK_BUCKET_SIZE> y;
	array<fixed32, SINK_BUCKET_SIZE> z;
	array<fixed32, KICK_PP_MAX> srcx;
	array<fixed32, KICK_PP_MAX> srcy;
	array<fixed32, KICK_PP_MAX> srcz;
};

__constant__ fmmpm_params dev_fmmpm_params;

__device__ inline int compute_indices(int index, int& total) {
	const int& tid = threadIdx.x;
	for (int P = 1; P < FMMPM_BLOCK_SIZE; P *= 2) {
		auto tmp = __shfl_up_sync(0xFFFFFFFF, index, P);
		if (tid >= P) {
			index += tmp;
		}
	}
	total = __shfl_sync(0xFFFFFFFF, index, FMMPM_BLOCK_SIZE - 1);
	auto tmp = __shfl_up_sync(0xFFFFFFFF, index, 1);
	if (tid >= 1) {
		index = tmp;
	} else {
		index = 0;
	}
	return index;
}

template<class T>
__device__ inline void shared_reduce(T& number) {
	for (int P = warpSize / 2; P >= 1; P /= 2) {
		number += __shfl_xor_sync(0xffffffff, number, P);
	}
}

__device__ void pp_interactions() {

}

__device__ void pc_interactions() {

}

__device__ void cp_interactions() {

}

__device__ void cc_interactions() {

}

__device__ void do_kick(checkitem mycheck) {
	const int& tid = threadIdx.x;
	const int& bid = blockIdx.x;
	const int& gsz = gridDim.x;
	__shared__ extern int shmem_ptr[];
	fmmpm_shmem& shmem = (fmmpm_shmem&) (*shmem_ptr);
	const auto& params = dev_fmmpm_params;
	auto& checklist = params.lists->checklist;
	auto& multilist = params.lists->multilist;
	auto& partlist = params.lists->partlist;
	auto& openlist = params.lists->openlist;
	auto& nextlist = params.lists->nextlist;
	const auto& theta2inv = 1.0f / sqr(params.theta);
	const bool iamleaf = mycheck.is_leaf();
	const float myradius = mycheck.get_radius();
	const fixed32 sink_x = mycheck.get_x(XDIM);
	const fixed32 sink_y = mycheck.get_x(YDIM);
	const fixed32 sink_z = mycheck.get_x(ZDIM);
	do {
		bool multi = false;
		bool part = false;
		bool open = false;
		bool next = false;
		for (int ci = tid; ci < checklist.size(); ci += warpSize) {
			auto& check = checklist[ci];
			const fixed32 src_x = check.get_x(XDIM);
			const fixed32 src_y = check.get_x(YDIM);
			const fixed32 src_z = check.get_x(ZDIM);
			const float source_radius = check.get_radius();
			const bool source_isleaf = check.is_leaf();
			const float dx = distance(sink_x, src_x);
			const float dy = distance(sink_y, src_y);
			const float dz = distance(sink_z, src_z);
			const float R2 = sqr(dx, dy, dz);
			const bool far = R2 > sqr(myradius + source_radius) * theta2inv;
			if (far) {
				if (!check.opened) {
					multi = true;
				} else {
					part = true;
				}
			} else {
				if (source_isleaf) {
					if (check.opened) {
						part = true;
					} else {
						open = true;
						check.opened = 1;
					}
				} else {
					next = true;
				}
			}
			int index, total;
			index = multi;
			index = compute_indices(index, total) + multilist.size();
			__syncwarp();
			if (tid == 0) {
				multilist.resize(multilist.size() + total);
			}
			__syncwarp();
			multilist[index] = check;
			index = part;
			index = compute_indices(index, total) + partlist.size();
			__syncwarp();
			if (tid == 0) {
				multilist.resize(partlist.size() + total);
			}
			__syncwarp();
			partlist[index] = check;
			index = next;
			index = compute_indices(index, total) + nextlist.size();
			__syncwarp();
			if (tid == 0) {
				nextlist.resize(partlist.size() + total);
			}
			__syncwarp();
			nextlist[index] = check;
			index = next;
			index = compute_indices(index, total) + openlist.size();
			__syncwarp();
			if (tid == 0) {
				openlist.resize(partlist.size() + total);
			}
			__syncwarp();
			openlist[index] = check;
		}
		__syncwarp();
		if (tid == 0) {
			checklist.resize(nextlist.size() * 2);
		}
		__syncwarp();
		for (int ci = 0; ci < nextlist.size(); ci += warpSize) {
			const auto children = nextlist[ci].get_children();
			checklist[2 * ci] = children[0];
			checklist[2 * ci + 1] = children[1];
		}
		const int offset = checklist.size();
		__syncwarp();
		if (tid == 0) {
			checklist.resize(offset + openlist.size());
		}
		__syncwarp();
		for (int ci = 0; ci < openlist.size(); ci += warpSize) {
			checklist[ci + offset] = openlist[ci];
		}
		if( mycheck.opened == 0 ) {
			cc_interactions();
			cp_interactions();
		} else {
			pc_interactions();
			pp_interactions();
		}
		__syncwarp();
		if (tid == 0) {
			nextlist.resize(0);
			openlist.resize(0);
			multilist.resize(0);
			partlist.resize(0);
		}
		__syncwarp();
		if (iamleaf) {
			mycheck.opened = 1;
		}
	} while (iamleaf && checklist.size());
}

__global__ void kick_fmmpm_kernel() {
	const int& tid = threadIdx.x;
	const int& bid = blockIdx.x;
	const int& gsz = gridDim.x;
	__shared__ extern int shmem_ptr[];
	fmmpm_shmem& shmem = (fmmpm_shmem&) (*shmem_ptr);
	const auto& params = dev_fmmpm_params;
	new (params.lists) list_set();
	auto& checklist = params.lists->checklist;
	const int cell_begin = size_t(bid) * (size_t) params.nsink_cells / (size_t) gsz;
	const int cell_end = size_t(bid + 1) * (size_t) params.nsink_cells / (size_t) gsz;
	for (int cell_index = cell_begin; cell_index < cell_end; cell_index++) {
		if (tid == 0) {
			checklist.resize(NCELLS);
		}
		__syncwarp();
		for (int treei = tid; treei < NCELLS; treei += warpSize) {
			checklist[treei].tr = params.tree_neighbors + cell_index * NCELLS + treei;
			checklist[treei].opened = 0;
			checklist[treei].index = 0;
		}
		checkitem mycheck;
		mycheck.tr = params.tree_neighbors + cell_index * NCELLS + NCELLS / 2;
		mycheck.opened = 0;
		mycheck.index = 0;
		__syncwarp();
		do_kick(mycheck);
	}

	params.lists->~list_set();
}

void kick_fmmpm(vector<tree> trees, range<int> box, int min_rung, double scale, double t0, bool first_call) {
	PRINT("shmem size = %i\n", sizeof(fmmpm_shmem));
//cudaFuncCache pCacheConfig;
	cudaDeviceSetCacheConfig (cudaFuncCachePreferShared);
//	cudaDeviceGetCacheConfig(&pCacheConfig);
	timer tm;
	size_t nsources = 0;
	size_t nsinks = 0;
	array<int, NDIM> i;
	const auto bigbox = box.pad(1);
	const size_t bigvol = bigbox.volume();
	const size_t vol = box.volume();
	int tree_size = 0;
	print("%i\n", bigvol);
	for (i[0] = bigbox.begin[0]; i[0] != bigbox.end[0]; i[0]++) {
		for (i[1] = bigbox.begin[1]; i[1] != bigbox.end[1]; i[1]++) {
			for (i[2] = bigbox.begin[2]; i[2] != bigbox.end[2]; i[2]++) {
				auto this_cell = chainmesh_get(i);
				nsources += this_cell.pend - this_cell.pbegin;
				const int index = bigbox.index(i);
				tree_size += trees[index].size() * sizeof(tree_node) + sizeof(tree);
			}
		}
	}
	PRINT("tree size = %e GB\n", tree_size / 1024 / 1024 / 1024.0);
	auto phibox = box;
	for (int dim = 0; dim < NDIM; dim++) {
		phibox.begin[dim] *= get_options().four_o_chain;
		phibox.end[dim] *= get_options().four_o_chain;
	}
	phibox = phibox.pad(PHI_BW);
	cudaFuncAttributes attr;
	cudaFuncGetAttributes(&attr, kick_fmmpm_kernel);
	if (attr.maxThreadsPerBlock < FMMPM_BLOCK_SIZE) {
		PRINT("This CUDA device will not run kick_pme_kernel with the required number of threads (%i)\n", FMMPM_BLOCK_SIZE);
		abort();
	}
	int occupancy;
	CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor ( &occupancy, kick_fmmpm_kernel,FMMPM_BLOCK_SIZE, sizeof(fmmpm_shmem)));
	PRINT("Occupancy = %i\n", occupancy);
	int num_blocks = 2 * occupancy * cuda_smp_count();
	const size_t mem_required = mem_requirements(nsources, nsinks, vol, bigvol, phibox.volume()) + tree_size + sizeof(fmmpm_params);
	const size_t free_mem = (size_t) 85 * cuda_free_mem() / size_t(100);
	PRINT("required = %li freemem = %li\n", mem_required, free_mem);
	if (mem_required > free_mem) {
		const auto child_boxes = box.split();
		PRINT("Splitting\n");
		kick_fmmpm(trees, child_boxes.first, min_rung, scale, t0, first_call);
		kick_fmmpm(std::move(trees), child_boxes.second, min_rung, scale, t0, first_call);
	} else {
		cuda_set_device();
		PRINT("Data transfer\n");
		tm.start();
		fmmpm_params params;
		params.allocate(nsources, nsinks, vol, bigvol, phibox.volume(), num_blocks);
		tm.stop();
		PRINT("%e\n", tm.read());
		tm.start();
		params.theta = 0.8;
		params.min_rung = min_rung;
		params.rs = get_options().rs;
		params.do_phi = true;
		params.rcut = 1.0 / get_options().chain_dim;
		params.hsoft = get_options().hsoft;
		params.phi0 = std::pow(get_options().parts_dim, NDIM) * 4.0 * M_PI * sqr(params.rs) - SELF_PHI / params.hsoft;
		PRINT("RCUT = %e RS\n", params.rcut / params.rs);
		params.GM = get_options().GM;
		params.Nfour = get_options().four_dim;
		params.phi_box = phibox;
		params.eta = get_options().eta;
		params.first_call = first_call;
		params.t0 = t0;
		params.scale = scale;
		params.inv2rs = 1.0f / params.rs / 2.0f;
		params.twooversqrtpi = 2.0f / sqrtf(M_PI);
		params.h2 = sqr(params.hsoft);
		params.hinv = 1.f / params.hsoft;
		params.h2inv = sqr(params.hinv);
		params.h3inv = params.hinv * sqr(params.hinv);
		tree* dev_tree_neighbors = (tree*) malloc(sizeof(tree) * NCELLS * vol);
		cudaStream_t stream;
		CUDA_CHECK(cudaStreamCreate(&stream));
		auto phi = gravity_long_get_phi(phibox);
		CUDA_CHECK(cudaMemcpyAsync(params.phi, phi.data(), sizeof(float) * phi.size(), cudaMemcpyHostToDevice, stream));

		struct cell_data {
			int box_index;
			int bigbox_index;
			chaincell cell;
		};
		vector<cell_data> chaincells;
		for (i[0] = bigbox.begin[0]; i[0] != bigbox.end[0]; i[0]++) {
			for (i[1] = bigbox.begin[1]; i[1] != bigbox.end[1]; i[1]++) {
				for (i[2] = bigbox.begin[2]; i[2] != bigbox.end[2]; i[2]++) {
					cell_data entry;
					entry.bigbox_index = bigbox.index(i);
					entry.cell = chainmesh_get(i);
					if (box.contains(i)) {
						const int q = box.index(i);
						entry.box_index = q;
					} else {
						entry.box_index = -1;
					}
					chaincells.push_back(entry);
				}
			}
		}
		std::sort(chaincells.begin(), chaincells.end(), [](cell_data a, cell_data b) {
			return a.cell.pbegin < b.cell.pbegin;
		});
		size_t count = 0;
		vector<cpymem> copies;
		for (int j = 0; j < chaincells.size(); j++) {
			auto this_cell = chaincells[j].cell;
			const auto this_size = this_cell.pend - this_cell.pbegin;
			const auto begin = this_cell.pbegin;
			const auto dif = count - begin;
			const int l = chaincells[j].bigbox_index;
			trees[l].adjust_indexes(dif);
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
			count += this_size;
		}

		vector<tree_node> all_trees;
		tree_node* dev_all_trees;
		count = 0;
		size_t trees_size = 0;
		for (int j = 0; j < bigvol; j++) {
			trees_size += trees[j].size();
		}
		all_trees.resize(trees_size);
		CUDA_CHECK(cudaMallocAsync(&dev_all_trees, sizeof(tree_node) * trees_size, stream));
		count = 0;
		vector<tree> dev_trees(bigvol);
		for (int j = 0; j < bigvol; j++) {
			dev_trees[j] = trees[j].to_device();
			dev_trees[j].nodes = dev_all_trees + count;
			std::memcpy(all_trees.data() + count, trees[j].nodes, sizeof(tree_node) * trees[j].size());
			count += dev_trees[j].size();
		}
		CUDA_CHECK(cudaMemcpyAsync(dev_all_trees, all_trees.data(), trees_size * sizeof(tree_node), cudaMemcpyHostToDevice));
		count = 0;
		for (i[0] = box.begin[0]; i[0] != box.end[0]; i[0]++) {
			for (i[1] = box.begin[1]; i[1] != box.end[1]; i[1]++) {
				for (i[2] = box.begin[2]; i[2] != box.end[2]; i[2]++) {
					auto this_cell = chainmesh_get(i);
					const auto this_size = this_cell.pend - this_cell.pbegin;
					const auto begin = this_cell.pbegin;
					cpymem cpy;
					const int l = box.index(i);
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
					count += this_size;
					array<int, NDIM> j;
					int p = 0;
					for (j[0] = i[0] - 1; j[0] <= i[0] + 1; j[0]++) {
						for (j[1] = i[1] - 1; j[1] <= i[1] + 1; j[1]++) {
							for (j[2] = i[2] - 1; j[2] <= i[2] + 1; j[2]++) {
								const int k = bigbox.index(j);
								std::memcpy(&dev_tree_neighbors[p + NCELLS * l], &dev_trees[k], sizeof(tree));
								p++;
							}
						}
					}
				}
			}
		}
		CUDA_CHECK(cudaMemcpyAsync(params.tree_neighbors, dev_tree_neighbors, sizeof(tree) * NCELLS * vol, cudaMemcpyHostToDevice, stream));
		process_copies(std::move(copies), cudaMemcpyHostToDevice, stream);
		CUDA_CHECK(cudaStreamSynchronize(stream));
		tm.stop();
		PRINT("Transfer time %e\n", tm.read());
		tm.reset();
		tm.start();
		PRINT("Launching kernel\n");
		CUDA_CHECK(cudaMemcpyToSymbol(dev_fmmpm_params, &params, sizeof(fmmpm_params)));
		kick_fmmpm_kernel<<<num_blocks,FMMPM_BLOCK_SIZE,sizeof(fmmpm_shmem),stream>>>();

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
		free(dev_tree_neighbors);
		CUDA_CHECK(cudaStreamDestroy(stream));
		CUDA_CHECK(cudaFree(dev_all_trees));
		tm.stop();
	}
}
