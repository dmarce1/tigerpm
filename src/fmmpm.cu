#include <tigerpm/fmmpm.hpp>
#include <tigerpm/timer.hpp>
#include <tigerpm/options.hpp>
#include <tigerpm/particles.hpp>
#include <tigerpm/util.hpp>
#include <tigerpm/gravity_long.hpp>
#include <tigerpm/gravity_short.hpp>
#include <tigerpm/stack_vector.hpp>
#include <tigerpm/kernels.hpp>

#include <thrust/device_vector.h>

#include <algorithm>

__constant__ float rung_dt[MAX_RUNG] = { 1.0 / (1 << 0), 1.0 / (1 << 1), 1.0 / (1 << 2), 1.0 / (1 << 3), 1.0 / (1 << 4), 1.0 / (1 << 5), 1.0 / (1 << 6), 1.0
		/ (1 << 7), 1.0 / (1 << 8), 1.0 / (1 << 9), 1.0 / (1 << 10), 1.0 / (1 << 11), 1.0 / (1 << 12), 1.0 / (1 << 13), 1.0 / (1 << 14), 1.0 / (1 << 15), 1.0
		/ (1 << 16), 1.0 / (1 << 17), 1.0 / (1 << 18), 1.0 / (1 << 19), 1.0 / (1 << 20), 1.0 / (1 << 21), 1.0 / (1 << 22), 1.0 / (1 << 23), 1.0 / (1 << 24), 1.0
		/ (1 << 25), 1.0 / (1 << 26), 1.0 / (1 << 27), 1.0 / (1 << 28), 1.0 / (1 << 29), 1.0 / (1 << 30), 1.0 / (1 << 31) };

#define LIST_SIZE  (8*1024)
#define STACK_SIZE (32*1024)
#define CUDA_STACK_SIZE (64*1024)
#define MAX_DEPTH 64

struct list_set;
struct checkitem;

struct fmmpm_params {
	fixed32* x;
	fixed32* y;
	fixed32* z;
	float* velx;
	float* vely;
	float* velz;
	float* phi;
	char* rung;
	kick_return* kreturn;
	list_set* lists;
	int* active;
	int* cell_indices;
	int min_rung;
	bool do_phi;
	float rs;
	float rcut;
	float rcut2;
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
	int* current_index;
	int nsink_cells;
	bool first_call;
	int Nfour;
	range<int> phi_box;
	int* tree_neighbors;
	tree_collection trees;
#ifdef FORCE_TEST
	float* gx;
	float* gy;
	float* gz;
	float* pot;
#endif
	void allocate(size_t source_size, size_t sink_size, size_t cell_count, size_t big_cell_count, size_t phi_cell_count, int nblocks);
	void free();
};

__constant__ fmmpm_params dev_fmmpm_params;

struct checkitem {
	int index;

	__device__ inline
	int get_nactive() const {
		return dev_fmmpm_params.trees.nodes[index].nactive;
	}
	__device__         inline fixed32 get_x(int dim) const {
		return dev_fmmpm_params.trees.nodes[index].multi.x[dim];
	}
	__device__ inline
	float get_radius() const {
		return dev_fmmpm_params.trees.nodes[index].radius;
	}
	__device__ inline
	int get_src_begin() const {
		return dev_fmmpm_params.trees.nodes[index].src_begin;
	}
	__device__ inline
	int get_src_end() const {
		return dev_fmmpm_params.trees.nodes[index].src_end;
	}
	__device__ inline
	int get_snk_begin() const {
		return dev_fmmpm_params.trees.nodes[index].snk_begin;
	}
	__device__ inline
	int get_snk_end() const {
		return dev_fmmpm_params.trees.nodes[index].snk_end;
	}
	__device__         inline multipole get_multipole() const {
		return dev_fmmpm_params.trees.nodes[index].multi.m;
	}
	__device__         inline array<checkitem, 2> get_children() {
		const auto indices = dev_fmmpm_params.trees.nodes[index].children;
		array<checkitem, 2> c;
		c[0].index = indices[0];
		c[1].index = indices[1];
		return c;
	}
};

struct list_set {
	stack_vector<checkitem, STACK_SIZE, MAX_DEPTH> checklist;
	fixedcapvec<checkitem, LIST_SIZE> leaflist;
	fixedcapvec<checkitem, LIST_SIZE> nextlist;
	fixedcapvec<checkitem, LIST_SIZE> multilist;
	fixedcapvec<checkitem, LIST_SIZE> pplist;
	fixedcapvec<checkitem, LIST_SIZE> cplist;
	array<expansion, MAX_DEPTH> Lexpansion;
};

void fmmpm_params::allocate(size_t source_size, size_t sink_size, size_t cell_count, size_t big_cell_count, size_t phi_cell_count, int nblocks) {
	CUDA_CHECK(cudaMalloc(&tree_neighbors, cell_count * NCELLS * sizeof(tree)));
	nsink_cells = cell_count;
	CUDA_CHECK(cudaMalloc(&current_index, sizeof(int)));
	int zero = 0;
	CUDA_CHECK(cudaMemcpy(current_index, &zero, sizeof(int), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMalloc(&cell_indices, cell_count * sizeof(int)));
	CUDA_CHECK(cudaMalloc(&x, source_size * sizeof(fixed32)));
	CUDA_CHECK(cudaMalloc(&y, source_size * sizeof(fixed32)));
	CUDA_CHECK(cudaMalloc(&z, source_size * sizeof(fixed32)));
	CUDA_CHECK(cudaMalloc(&lists, nblocks * sizeof(list_set)));
	CUDA_CHECK(cudaMalloc(&phi, phi_cell_count * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&active, nblocks * sizeof(int) * SINK_BUCKET_SIZE));
}
void fmmpm_params::free() {
	CUDA_CHECK(cudaFree(current_index));
	CUDA_CHECK(cudaFree(cell_indices));
	CUDA_CHECK(cudaFree(x));
	CUDA_CHECK(cudaFree(y));
	CUDA_CHECK(cudaFree(z));
	CUDA_CHECK(cudaFree(phi));
	CUDA_CHECK(cudaFree(active));
	CUDA_CHECK(cudaFree(lists));
	CUDA_CHECK(cudaFree(tree_neighbors));
}

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
	//PRINT("Compressed from %i to %i copies\n", copies.size(), compressed.size());
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

__device__ inline int compute_indices(int index, int& total) {
	const int& tid = threadIdx.x;
	for (int P = 1; P < WARP_SIZE; P *= 2) {
		auto tmp = __shfl_up_sync(0xFFFFFFFF, index, P);
		if (tid >= P) {
			index += tmp;
		}
	}
	total = __shfl_sync(0xFFFFFFFF, index, WARP_SIZE - 1);
	auto tmp = __shfl_up_sync(0xFFFFFFFF, index, 1);
	if (tid >= 1) {
		index = tmp;
	} else {
		index = 0;
	}
	return index;
}

template<class T>
__device__ inline void shared_reduce_add(T& number) {
	for (int P = warpSize / 2; P >= 1; P /= 2) {
		number += __shfl_xor_sync(0xffffffff, number, P);
	}
}

__device__ inline void shared_reduce_min(float& number) {
	for (int P = warpSize / 2; P >= 1; P /= 2) {
		number = fminf(number, __shfl_xor_sync(0xffffffff, number, P));
	}
}

__device__ inline void shared_reduce_max(int& number) {
	for (int P = warpSize / 2; P >= 1; P /= 2) {
		number = max(number, __shfl_xor_sync(0xffffffff, number, P));
	}
}

inline __device__ int compute_pp_interaction(float dx, float dy, float dz, float& gx, float& gy, float& gz, float& phi) {
	int flops = 0;
	const fmmpm_params& params = dev_fmmpm_params;
	const float& twooversqrtpi = params.twooversqrtpi;
	const float& h2 = params.h2;
	const float& hinv = params.hinv;
	const float& h2inv = params.h2inv;
	const float& h3inv = params.h3inv;
	const float& inv2rs = params.inv2rs;
	const float& rcut2 = params.rcut2;
	float rinv, rinv3;
	const float r2 = sqr(dx, dy, dz);
	flops += 1;
	if (r2 < rcut2) {
		float exp0;
		if (r2 > 0.0f) {
			rinv = rsqrtf(r2);
			flops += 4;
		} else {
			rinv = 0.f;
		}
		const float r = r2 * rinv;
		const float r0 = r * inv2rs;
		const float erfc0 = erfcexp(r0, &exp0);
		if (r2 > h2) {
			rinv3 = (erfc0 + twooversqrtpi * r0 * exp0) * rinv * rinv * rinv;
			rinv *= erfc0;
			flops += 7;
		} else {
			const float q2 = r2 * h2inv;
			float d1 = +15.0f / 8.0f;
			d1 = fmaf(d1, q2, -21.0f / 4.0f);
			d1 = fmaf(d1, q2, +35.0f / 8.0f);
			d1 *= h3inv;
			rinv3 = ((erfc0 - 1.0f) + twooversqrtpi * r0 * exp0) * rinv * rinv * rinv + d1;
			flops += 14;
			if (params.do_phi) {
				float d0 = -5.0f / 16.0f;
				d0 = fmaf(d0, q2, 21.0f / 16.0f);
				d0 = fmaf(d0, q2, -35.0f / 16.0f);
				d0 = fmaf(d0, q2, 35.0f / 16.0f);
				d0 *= hinv;
				rinv = (erfc0 - 1.0f) * rinv + d0;
				flops += 10;
			}
		}
		gx -= dx * rinv3;
		gy -= dy * rinv3;
		gz -= dz * rinv3;
		phi -= rinv;
		flops += 42;
	}
	return flops;
}

__device__ int pp_interactions(int nactive) {
	int flops = 0;
	const int& tid = threadIdx.x;
	const int& bid = blockIdx.x;
	__shared__ extern int shmem_ptr[];
	fmmpm_shmem& shmem = (fmmpm_shmem&) (*shmem_ptr);
	const fmmpm_params& params = dev_fmmpm_params;
	const auto& list = (params.lists + bid)->pplist;
	int i = 0;
	int N = 0;
	int part_index;
	if (list.size() == 0) {
		return 0;
	}
	int n = 0;
	auto these_parts_begin = list[i].get_src_begin();
	auto these_parts_end = list[i].get_src_end();
	while (i < list.size()) {
		__syncwarp();
		part_index = 0;
		while (part_index < KICK_PP_MAX && i < list.size()) {
			while (i + 1 < list.size()) {
				const auto next_parts_begin = list[i + 1].get_src_begin();
				const auto next_parts_end = list[i + 1].get_src_end();
				if (these_parts_end == next_parts_begin) {
					these_parts_end = next_parts_end;
					i++;
				} else {
					break;
				}
			}
			const int imin = these_parts_begin;
			const int imax = min(these_parts_begin + (KICK_PP_MAX - part_index), these_parts_end);
			const int sz = imax - imin;
			for (int j = tid; j < sz; j += warpSize) {
				for (int dim = 0; dim < NDIM; dim++) {
					shmem.srcx[part_index + j] = params.x[j + imin];
					shmem.srcy[part_index + j] = params.y[j + imin];
					shmem.srcz[part_index + j] = params.z[j + imin];
				}
			}
			these_parts_begin += sz;
			part_index += sz;
			if (these_parts_begin == these_parts_end) {
				i++;
				if (i < list.size()) {
					these_parts_begin = list[i].get_src_begin();
					these_parts_end = list[i].get_src_end();
				}
			}
		}
		int mid_index;
		if ((nactive % warpSize) < warpSize / 2) {
			mid_index = nactive - (nactive % warpSize);
		} else {
			mid_index = nactive;
		}
		__syncwarp();
		for (int sink_index = tid; sink_index < mid_index; sink_index += warpSize) {
			float& phi = shmem.phi[sink_index];
			auto& g = shmem.g[sink_index];
			const fixed32& sink_x = shmem.x[sink_index];
			const fixed32& sink_y = shmem.y[sink_index];
			const fixed32& sink_z = shmem.z[sink_index];
			N += part_index;
			for (int j = 0; j < part_index; j++) {
				const fixed32& src_x = shmem.srcx[j];
				const fixed32& src_y = shmem.srcy[j];
				const fixed32& src_z = shmem.srcz[j];
				const float dx = distance(sink_x, src_x);
				const float dy = distance(sink_y, src_y);
				const float dz = distance(sink_z, src_z);
				flops += 3;
				flops += compute_pp_interaction(dx, dy, dz, g[XDIM], g[YDIM], g[ZDIM], phi);
				n++;
			}
		}
		__syncwarp();
		for (int sink_index = mid_index; sink_index < nactive; sink_index++) {
			float phi = 0.0f;
			array<float, NDIM> g;
			g[XDIM] = g[YDIM] = g[ZDIM] = 0.f;
			const fixed32& sink_x = shmem.x[sink_index];
			const fixed32& sink_y = shmem.y[sink_index];
			const fixed32& sink_z = shmem.z[sink_index];
			N += part_index;
			for (int j = tid; j < part_index; j += warpSize) {
				const fixed32& src_x = shmem.srcx[j];
				const fixed32& src_y = shmem.srcy[j];
				const fixed32& src_z = shmem.srcz[j];
				const float dx = distance(sink_x, src_x);
				const float dy = distance(sink_y, src_y);
				const float dz = distance(sink_z, src_z);
				flops += 3;
				flops += compute_pp_interaction(dx, dy, dz, g[XDIM], g[YDIM], g[ZDIM], phi);
				n++;
			}
			for (int dim = 0; dim < NDIM; dim++) {
				shared_reduce_add(g[dim]);
			}
			shared_reduce_add(phi);
			if (tid == 0) {
				for (int dim = 0; dim < NDIM; dim++) {
					shmem.g[sink_index][dim] += g[dim];
				}
				if (params.do_phi) {
					shmem.phi[sink_index] += phi;
				}
			}
			__syncwarp();
		}
	}
	shared_reduce_add(n);
	if( tid == 0 ) {
		atomicAdd(&params.kreturn->pp, (double) n);
	}
	return flops;
}

__device__ int pc_interactions(int nactive) {
	int flops = 0;
	const int& tid = threadIdx.x;
	const int& bid = blockIdx.x;
	__shared__ extern int shmem_ptr[];
	fmmpm_shmem& shmem = (fmmpm_shmem&) (*shmem_ptr);
	const fmmpm_params& params = dev_fmmpm_params;
	const auto& list = (params.lists + bid)->multilist;
	int n = 0;
	int kmid;
	if (nactive % warpSize < warpSize / 8) {
		kmid = nactive - nactive % warpSize;
	} else {
		kmid = nactive;
	}
	for (int k = tid; k < kmid; k += warpSize) {
		float& phi = shmem.phi[k];
		auto& g = shmem.g[k];
		const fixed32 sink_x = shmem.x[k];
		const fixed32 sink_y = shmem.y[k];
		const fixed32 sink_z = shmem.z[k];
		array<float, NDIM + 1> L;
		L[0] = L[1] = L[2] = L[3] = 0.f;
		for (int i = 0; i < list.size(); i++) {
			const fixed32 src_x = list[i].get_x(XDIM);
			const fixed32 src_y = list[i].get_x(YDIM);
			const fixed32 src_z = list[i].get_x(ZDIM);
			const float dx = distance(sink_x, src_x);
			const float dy = distance(sink_y, src_y);
			const float dz = distance(sink_z, src_z);
			flops += 3;
			const auto M = list[i].get_multipole();
			expansion D;
			for (int j = 0; j < EXPANSION_SIZE; j++) {
				D[j] = 0.f;
			}
			flops += greens_function(D, dx, dy, dz, params.inv2rs);
			flops += M2L_kernel(L, M, D, params.do_phi);
			n++;
		}
		g[XDIM] -= L[XDIM + 1];
		g[YDIM] -= L[YDIM + 1];
		g[ZDIM] -= L[ZDIM + 1];
		phi += L[0];
	}
	for (int k = kmid; k < nactive; k++) {
		float& phi = shmem.phi[k];
		auto& g = shmem.g[k];
		const fixed32 sink_x = shmem.x[k];
		const fixed32 sink_y = shmem.y[k];
		const fixed32 sink_z = shmem.z[k];
		array<float, NDIM + 1> L;
		L[0] = L[1] = L[2] = L[3] = 0.f;
		for (int i = tid; i < list.size(); i += warpSize) {
			const fixed32 src_x = list[i].get_x(XDIM);
			const fixed32 src_y = list[i].get_x(YDIM);
			const fixed32 src_z = list[i].get_x(ZDIM);
			const float dx = distance(sink_x, src_x);
			const float dy = distance(sink_y, src_y);
			const float dz = distance(sink_z, src_z);
			flops += 3;
			const auto M = list[i].get_multipole();
			expansion D;
			for (int j = 0; j < EXPANSION_SIZE; j++) {
				D[j] = 0.f;
			}
			flops += greens_function(D, dx, dy, dz, params.inv2rs);
			flops += M2L_kernel(L, M, D, params.do_phi);
			n++;
		}
		for (int P = warpSize / 2; P >= 1; P /= 2) {
			for (int i = 0; i < NDIM + 1; i++) {
				L[i] += __shfl_xor_sync(0xffffffff, L[i], P);
			}
		}
		if (tid == 0) {
			g[XDIM] -= L[XDIM + 1];
			g[YDIM] -= L[YDIM + 1];
			g[ZDIM] -= L[ZDIM + 1];
			phi += L[0];
		}
	}
	__syncwarp();
	shared_reduce_add(n);
	if( tid == 0 ) {
		atomicAdd(&params.kreturn->pc, (double) n);
	}
	return flops;
}

__device__ int cp_interactions(checkitem mycheck, expansion& Lexpansion) {
	int flops = 0;
	const int& tid = threadIdx.x;
	const int& bid = blockIdx.x;
	__shared__ extern int shmem_ptr[];
	fmmpm_shmem& shmem = (fmmpm_shmem&) (*shmem_ptr);
	const fmmpm_params& params = dev_fmmpm_params;
	const auto& list = (params.lists + bid)->cplist;
	int i = 0;
	int N = 0;
	int n = 0;
	int part_index;
	if (list.size() == 0) {
		return 0;
	}
	auto these_parts_begin = list[i].get_src_begin();
	auto these_parts_end = list[i].get_src_end();
	while (i < list.size()) {
		__syncwarp();
		part_index = 0;
		while (part_index < KICK_PP_MAX && i < list.size()) {
			while (i + 1 < list.size()) {
				const auto next_parts_begin = list[i + 1].get_src_begin();
				const auto next_parts_end = list[i + 1].get_src_end();
				if (these_parts_end == next_parts_begin) {
					these_parts_end = next_parts_end;
					i++;
				} else {
					break;
				}
			}
			const int imin = these_parts_begin;
			const int imax = min(these_parts_begin + (KICK_PP_MAX - part_index), these_parts_end);
			const int sz = imax - imin;
			for (int j = tid; j < sz; j += warpSize) {
				for (int dim = 0; dim < NDIM; dim++) {
					shmem.srcx[part_index + j] = params.x[j + imin];
					shmem.srcy[part_index + j] = params.y[j + imin];
					shmem.srcz[part_index + j] = params.z[j + imin];
				}
			}
			these_parts_begin += sz;
			part_index += sz;
			if (these_parts_begin == these_parts_end) {
				i++;
				if (i < list.size()) {
					these_parts_begin = list[i].get_src_begin();
					these_parts_end = list[i].get_src_end();
				}
			}
		}
		__syncwarp();
		const fixed32 sink_x = mycheck.get_x(XDIM);
		const fixed32 sink_y = mycheck.get_x(YDIM);
		const fixed32 sink_z = mycheck.get_x(ZDIM);
		N += part_index;
		expansion L;
		for (int i = 0; i < EXPANSION_SIZE; i++) {
			L[i] = 0.f;
		}
		for (int j = tid; j < part_index; j += warpSize) {
			const fixed32& src_x = shmem.srcx[j];
			const fixed32& src_y = shmem.srcy[j];
			const fixed32& src_z = shmem.srcz[j];
			const float dx = distance(sink_x, src_x);
			const float dy = distance(sink_y, src_y);
			const float dz = distance(sink_z, src_z);
			flops += 3;
			flops += greens_function(L, dx, dy, dz, params.inv2rs);
			n++;
		}
		for (int P = warpSize / 2; P >= 1; P /= 2) {
			for (int i = 0; i < EXPANSION_SIZE; i++) {
				L[i] += __shfl_xor_sync(0xffffffff, L[i], P);
			}
		}
		for (int i = tid; i < EXPANSION_SIZE; i += warpSize) {
			Lexpansion[i] += L[i];
		}
	}
	shared_reduce_add(n);
	if( tid == 0 ) {
		atomicAdd(&params.kreturn->cp, (double) n);
	}
	return flops;
}

__device__ int cc_interactions(checkitem mycheck, expansion& Lexpansion) {
	int flops = 0;
	const int& tid = threadIdx.x;
	const int& bid = blockIdx.x;
	__shared__ extern int shmem_ptr[];
	const fmmpm_params& params = dev_fmmpm_params;
	const auto& list = (params.lists + bid)->multilist;
	int n = 0;
	expansion L;
	const fixed32 sink_x = mycheck.get_x(XDIM);
	const fixed32 sink_y = mycheck.get_x(YDIM);
	const fixed32 sink_z = mycheck.get_x(ZDIM);
	for (int i = 0; i < EXPANSION_SIZE; i++) {
		L[i] = 0.f;
	}
	for (int i = tid; i < list.size(); i += warpSize) {
		const fixed32 src_x = list[i].get_x(XDIM);
		const fixed32 src_y = list[i].get_x(YDIM);
		const fixed32 src_z = list[i].get_x(ZDIM);
		const float dx = distance(sink_x, src_x);
		const float dy = distance(sink_y, src_y);
		const float dz = distance(sink_z, src_z);
		flops += 3;
		const auto M = list[i].get_multipole();
		expansion D;
		for (int j = 0; j < EXPANSION_SIZE; j++) {
			D[j] = 0.f;
		}
		flops += greens_function(D, dx, dy, dz, params.inv2rs);
		flops += M2L_kernel(L, M, D, params.do_phi);
		n++;
	}
	for (int P = warpSize / 2; P >= 1; P /= 2) {
		for (int i = 0; i < EXPANSION_SIZE; i++) {
			L[i] += __shfl_xor_sync(0xffffffff, L[i], P);
		}
	}
	for (int i = tid; i < EXPANSION_SIZE; i += warpSize) {
		Lexpansion[i] += L[i];
	}
	__syncwarp();
	shared_reduce_add(n);
	if( tid == 0 ) {
		atomicAdd(&params.kreturn->cc, (double) n);
	}
	return flops;
}

__device__ int long_range_interp(int nactive) {
	const int& tid = threadIdx.x;
	__shared__ extern int shmem_ptr[];
	fmmpm_shmem& shmem = (fmmpm_shmem&) (*shmem_ptr);
	const auto& params = dev_fmmpm_params;
	for (int sink_index = tid; sink_index < nactive; sink_index += TREEPM_BLOCK_SIZE) {
		array<float, NDIM>& g = shmem.g[sink_index];
		float& phi = shmem.phi[sink_index];
		g[0] = g[1] = g[2] = 0.f;
		phi = params.phi0;
		const fixed32 sink_x = shmem.x[sink_index];
		const fixed32 sink_y = shmem.y[sink_index];
		const fixed32 sink_z = shmem.z[sink_index];
		array<int, NDIM> I;
		array<int, NDIM> J;
		array<float, NDIM> X;
		array<array<float, 4>, NDIM> w;
		X[XDIM] = sink_x.to_float();
		X[YDIM] = sink_y.to_float();
		X[ZDIM] = sink_z.to_float();
		array<array<array<array<float, 4>, 4>, 4>, NDIM> force;
		array<array<array<float, 8>, 8>, 8> pot;
		for (int dim = 0; dim < NDIM; dim++) {
			X[dim] *= params.Nfour;
			I[dim] = min(int(X[dim]), params.phi_box.end[dim] - PHI_BW);
			I[dim] -= 3;
		}
		for (J[0] = I[0]; J[0] < I[0] + 8; J[0]++) {
			for (J[1] = I[1]; J[1] < I[1] + 8; J[1]++) {
				for (J[2] = I[2]; J[2] < I[2] + 8; J[2]++) {
					const int l = params.phi_box.index(J);
					pot[J[0] - I[0]][J[1] - I[1]][J[2] - I[2]] = params.phi[l];
				}
			}
		}
		for (J[0] = I[0] + 2; J[0] < I[0] + 6; J[0]++) {
			for (J[1] = I[1] + 2; J[1] < I[1] + 6; J[1]++) {
				for (J[2] = I[2] + 2; J[2] < I[2] + 6; J[2]++) {
					const int i = J[0] - I[0] - 2;
					const int j = J[1] - I[1] - 2;
					const int k = J[2] - I[2] - 2;
					const int ip = i + 2;
					const int jp = j + 2;
					const int kp = k + 2;
					force[0][i][j][k] = -((2.f / 3.f) * (pot[ip + 1][jp][kp] - pot[ip - 1][jp][kp]) - (1.f / 12.f) * (pot[ip + 2][jp][kp] - pot[ip - 2][jp][kp]))
							* params.Nfour;
					force[1][i][j][k] = -((2.f / 3.f) * (pot[ip][jp + 1][kp] - pot[ip][jp - 1][kp]) - (1.f / 12.f) * (pot[ip][jp + 2][kp] - pot[ip][jp - 2][kp]))
							* params.Nfour;
					force[2][i][j][k] = -((2.f / 3.f) * (pot[ip][jp][kp + 1] - pot[ip][jp][kp - 1]) - (1.f / 12.f) * (pot[ip][jp][kp + 2] - pot[ip][jp][kp - 2]))
							* params.Nfour;
				}
			}
		}
		for (int dim = 0; dim < NDIM; dim++) {
			for (int i = 0; i < 4; i++) {
				const float x = X[dim] - (I[dim] + 2 + i);
				w[dim][i] = cloud4(x);
			}
		}
		float gx = 0.f;
		for (J[0] = I[0] + 2; J[0] < I[0] + 6; J[0]++) {
			for (J[1] = I[1] + 2; J[1] < I[1] + 6; J[1]++) {
				for (J[2] = I[2] + 2; J[2] < I[2] + 6; J[2]++) {
					const int i = J[0] - I[0] - 2;
					const int j = J[1] - I[1] - 2;
					const int k = J[2] - I[2] - 2;
					gx += force[0][i][j][k] * w[0][i] * w[1][j] * w[2][k];
				}
			}
		}
		float gy = 0.f;
		for (J[0] = I[0] + 2; J[0] < I[0] + 6; J[0]++) {
			for (J[1] = I[1] + 2; J[1] < I[1] + 6; J[1]++) {
				for (J[2] = I[2] + 2; J[2] < I[2] + 6; J[2]++) {
					const int i = J[0] - I[0] - 2;
					const int j = J[1] - I[1] - 2;
					const int k = J[2] - I[2] - 2;
					gy += force[1][i][j][k] * w[0][i] * w[1][j] * w[2][k];
				}
			}
		}
		float gz = 0.f;
		for (J[0] = I[0] + 2; J[0] < I[0] + 6; J[0]++) {
			for (J[1] = I[1] + 2; J[1] < I[1] + 6; J[1]++) {
				for (J[2] = I[2] + 2; J[2] < I[2] + 6; J[2]++) {
					const int i = J[0] - I[0] - 2;
					const int j = J[1] - I[1] - 2;
					const int k = J[2] - I[2] - 2;
					gz += force[2][i][j][k] * w[0][i] * w[1][j] * w[2][k];
				}
			}
		}
		if (params.do_phi) {
			for (J[0] = I[0] + 2; J[0] < I[0] + 6; J[0]++) {
				for (J[1] = I[1] + 2; J[1] < I[1] + 6; J[1]++) {
					for (J[2] = I[2] + 2; J[2] < I[2] + 6; J[2]++) {
						const int i = J[0] - I[0];
						const int j = J[1] - I[1];
						const int k = J[2] - I[2];
						phi += pot[i][j][k] * w[0][i - 2] * w[1][j - 2] * w[2][k - 2];
					}
				}
			}
		}
		g[XDIM] += gx;
		g[YDIM] += gy;
		g[ZDIM] += gz;
	}
	__syncwarp();
	return 0;
}
__device__ void do_kick(size_t stack, checkitem mycheck, int depth, array<fixed32, NDIM> Lpos) {
	if (depth >= MAX_DEPTH) {
		PRINT("MAX_DEPTH exceeded!\n");
		__trap();
	}
	int flops = 0;
//	auto tm = clock64();
	const int& tid = threadIdx.x;
	const int& bid = blockIdx.x;
	__shared__ extern int shmem_ptr[];
	fmmpm_shmem& shmem = (fmmpm_shmem&) (*shmem_ptr);
	const auto& params = dev_fmmpm_params;
	const auto& theta2inv = 1.0f / sqr(params.theta);
	const auto& thetainv = 1.0f / params.theta;
	auto* lists = params.lists + bid;
	auto& checklist = lists->checklist;
	auto& multilist = lists->multilist;
	auto& pplist = lists->pplist;
	auto& cplist = lists->cplist;
	auto& leaflist = lists->leaflist;
	auto& nextlist = lists->nextlist;
	auto& Lexpansion = lists->Lexpansion[depth];
	auto* active = params.active + bid * SINK_BUCKET_SIZE;
	const bool iamleaf = (mycheck.get_snk_end() - mycheck.get_snk_begin()) <= SINK_BUCKET_SIZE;
	const float myradius = mycheck.get_radius();
	const fixed32 sink_x = mycheck.get_x(XDIM);
	const fixed32 sink_y = mycheck.get_x(YDIM);
	const fixed32 sink_z = mycheck.get_x(ZDIM);
	array<float, NDIM> dX;
	dX[XDIM] = distance(sink_x, Lpos[XDIM]);
	dX[YDIM] = distance(sink_y, Lpos[YDIM]);
	dX[ZDIM] = distance(sink_z, Lpos[ZDIM]);
	auto Ltmp = L2L_kernel(Lexpansion, dX, params.do_phi);
	__syncwarp();
	for (int i = tid; i < EXPANSION_SIZE; i += warpSize) {
		Lexpansion[i] = Ltmp[i];
	}
//	atomicAdd(&Ltime, (double) (clock64() - tm));
//	tm = clock64();
	if (tid == 0) {
		flops += 1456 + params.do_phi * 220;
		multilist.resize(0);
		leaflist.resize(0);
		cplist.resize(0);
	}
	__syncwarp();
	do {
		__syncwarp();
		if (tid == 0) {
			nextlist.resize(0);
		}
		__syncwarp();
		const int maxi = round_up(checklist.size(), warpSize);
		bool multi;
		bool next;
		bool leaf;
		bool cp;
		for (int ci = tid; ci < maxi; ci += warpSize) {
			checkitem check;
			multi = false;
			leaf = false;
			next = false;
			cp = false;
			if (ci < checklist.size()) {
				check = checklist[ci];
				const bool source_isleaf = (check.get_src_end() - check.get_src_begin()) <= SOURCE_BUCKET_SIZE;
				const fixed32 src_x = check.get_x(XDIM);
				const fixed32 src_y = check.get_x(YDIM);
				const fixed32 src_z = check.get_x(ZDIM);
				const float source_radius = check.get_radius();
				const float dx = distance(sink_x, src_x);
				const float dy = distance(sink_y, src_y);
				const float dz = distance(sink_z, src_z);
				const float R2 = sqr(dx, dy, dz);
				const bool veryfar = R2 > sqr(myradius + source_radius + params.rcut);
				const bool far1 = R2 > sqr(myradius * 1.5 + source_radius) * theta2inv;
				const bool far2 = R2 > sqr(myradius * 1.5 * thetainv + source_radius);
				if (!veryfar) {
					if (far1) {
						multi = true;
					} else if (far2 &&  source_isleaf) {
						cp = true;
					} else if (!source_isleaf) {
						next = true;
					} else {
						leaf = true;
					}
				}
				flops += 19;
			}
			int index, total;

			index = multi;
			index = compute_indices(index, total) + multilist.size();
			__syncwarp();
			if (tid == 0) {
				multilist.resize(multilist.size() + total);
			}
			__syncwarp();
			if (multi) {
				multilist[index] = check;
			}

			index = cp;
			index = compute_indices(index, total) + cplist.size();
			__syncwarp();
			if (tid == 0) {
				cplist.resize(cplist.size() + total);
			}
			__syncwarp();
			if (cp) {
				cplist[index] = check;
			}

			index = next;
			index = compute_indices(index, total) + nextlist.size();
			__syncwarp();
			if (tid == 0) {
				nextlist.resize(nextlist.size() + total);
			}
			__syncwarp();
			if (next) {
				nextlist[index] = check;
			}

			index = leaf;
			index = compute_indices(index, total) + leaflist.size();
			__syncwarp();
			if (tid == 0) {
				leaflist.resize(leaflist.size() + total);
			}
			__syncwarp();
			if (leaf) {
				leaflist[index] = check;
			}
		}

		__syncwarp();
		if (tid == 0) {
			checklist.resize(nextlist.size() * NCHILD);
		}
		__syncwarp();
		for (int ci = tid; ci < nextlist.size(); ci += warpSize) {
			const auto children = nextlist[ci].get_children();
			for (int i = 0; i < NCHILD; i++) {
				checklist[2 * ci + i] = children[i];
			}
		}

		if (!iamleaf) {
			const int offset = checklist.size();
			__syncwarp();
			if (tid == 0) {
				checklist.resize(offset + leaflist.size());
			}
			__syncwarp();
			for (int ci = tid; ci < leaflist.size(); ci += warpSize) {
				checklist[ci + offset] = leaflist[ci];
			}
			__syncwarp();
			if (tid == 0) {
				leaflist.resize(0);
			}
			__syncwarp();
		}
		__syncwarp();

	} while (iamleaf && checklist.size());
//	atomicAdd(&chk1time, (double) (clock64() - tm));
//	tm = clock64();
	flops += cc_interactions(mycheck, Lexpansion);
	flops += cp_interactions(mycheck, Lexpansion);
//	atomicAdd(&cctime, (double) (clock64() - tm));

	if (iamleaf) {
//		tm = clock64();
		const auto snk_begin = mycheck.get_snk_begin();
		const auto snk_end = mycheck.get_snk_end();
		const int nsinks = snk_end - snk_begin;
		const int imax = round_up(nsinks, warpSize);
		int nactive = 0;
		for (int i = tid; i < imax; i += warpSize) {
			const int snki = snk_begin + i;
			bool is_active;
			if (i < nsinks) {
				is_active = params.do_phi || (params.rung[snki] >= params.min_rung);
			} else {
				is_active = false;
			}
			int total;
			int active_index = compute_indices(int(is_active), total) + nactive;
			int srci = mycheck.get_src_begin() + i;
			if (is_active) {
				active[active_index] = snki;
				shmem.x[active_index] = params.x[srci];
				shmem.y[active_index] = params.y[srci];
				shmem.z[active_index] = params.z[srci];
			}
			nactive += total;
			__syncwarp();
		}
//		atomicAdd(&acttime, (double) (clock64() - tm));
//		tm = clock64();

		flops += long_range_interp(nactive);
//		atomicAdd(&longtime, (double) (clock64() - tm));
//		tm = clock64();
		__syncwarp();
		if (tid == 0) {
			pplist.resize(0);
			cplist.resize(0);
			multilist.resize(0);
		}
		__syncwarp();
		const int maxi = round_up(leaflist.size(), warpSize);
		for (int i = tid; i < maxi; i += warpSize) {
			bool pc = false;
			bool pp = false;
			checkitem check;
			if (i < leaflist.size()) {
				check = leaflist[i];
				const int begin = mycheck.get_src_begin();
				const int end = mycheck.get_src_end();
				const fixed32 src_x = check.get_x(XDIM);
				const fixed32 src_y = check.get_x(YDIM);
				const fixed32 src_z = check.get_x(ZDIM);
				float source_radius = check.get_radius();
				bool far = true;
				for (int j = 0; j < nactive; j++) {
					const fixed32& sink_x = shmem.x[j];
					const fixed32& sink_y = shmem.y[j];
					const fixed32& sink_z = shmem.z[j];
					const float dx = distance(sink_x, src_x);
					const float dy = distance(sink_y, src_y);
					const float dz = distance(sink_z, src_z);
					const float R2 = sqr(dx, dy, dz);
					far = R2 > sqr(source_radius) * theta2inv;
					flops += 11;
					if (!far) {
						break;
					}
				}
				pp = !far;
				pc = far;
			}
			int total;
			int index = pp;
			index = compute_indices(index, total) + pplist.size();
			__syncwarp();
			if (tid == 0) {
				pplist.resize(pplist.size() + total);
			}
			__syncwarp();
			if (pp) {
				pplist[index] = check;
			}
			index = pc;
			index = compute_indices(index, total) + multilist.size();
			__syncwarp();
			if (tid == 0) {
				multilist.resize(multilist.size() + total);
			}
			__syncwarp();
			if (pc) {
				multilist[index] = check;
			}

		}
//		atomicAdd(&chk2time, (double) (clock64() - tm));
//		tm = clock64();

		flops += pc_interactions(nactive);
//		atomicAdd(&pctime, (double) (clock64() - tm));
//		tm = clock64();
		flops += pp_interactions(nactive);
//		atomicAdd(&pptime, (double) (clock64() - tm));
//		tm = clock64();
		int max_rung = -1;
		float pot = 0.0f;
		float fx = 0.f;
		float fy = 0.f;
		float fz = 0.f;
		float fnorm = 0.f;
//		PRINT( "%li\n", (size_t) &stack - stack);
		for (int sink_index = tid; sink_index < nactive; sink_index += warpSize) {
			array<float, NDIM>& g = shmem.g[sink_index];
			float& phi = shmem.phi[sink_index];
			dX[XDIM] = distance(shmem.x[sink_index], sink_x);
			dX[YDIM] = distance(shmem.y[sink_index], sink_y);
			dX[ZDIM] = distance(shmem.z[sink_index], sink_z);
			const auto L2 = L2P_kernel(Lexpansion, dX, params.do_phi);
			phi += L2[0];
			g[XDIM] -= L2[XDIM + 1];
			g[YDIM] -= L2[YDIM + 1];
			g[ZDIM] -= L2[ZDIM + 1];
			const int snki = active[sink_index];
			g[XDIM] *= params.GM;
			g[YDIM] *= params.GM;
			g[ZDIM] *= params.GM;
			phi *= params.GM;
			pot += phi;
			fx += g[XDIM];
			fy += g[YDIM];
			fz += g[ZDIM];
			fnorm += sqr(g[XDIM], g[YDIM], g[ZDIM]);
#ifdef FORCE_TEST
			params.gx[snki] = g[XDIM];
			params.gy[snki] = g[YDIM];
			params.gz[snki] = g[ZDIM];
			params.pot[snki] = phi;
#endif
			flops += 9;
			if (params.rung[snki] >= params.min_rung) {
				auto& vx = params.velx[snki];
				auto& vy = params.vely[snki];
				auto& vz = params.velz[snki];
				auto& rung = params.rung[snki];
				auto dt = 0.5f * rung_dt[rung] * params.t0;
				if (!params.first_call) {
					vx = fmaf(g[XDIM], dt, vx);
					vy = fmaf(g[YDIM], dt, vy);
					vz = fmaf(g[ZDIM], dt, vz);
					flops += 6;
				}
				flops += 39 + 337 + params.do_phi * 158;
				const auto g2 = sqr(g[0], g[1], g[2]);
				const auto factor = params.eta * sqrtf(params.scale * params.hsoft);
				dt = fminf(factor * rsqrt(sqrtf(g2)), params.t0);
				rung = max((int) ceilf(log2f(params.t0) - log2f(dt)), max(rung - 1, params.min_rung));
				max_rung = max(rung, max_rung);
				if (rung < 0 || rung >= MAX_RUNG) {
					PRINT("Rung out of range %i\n", rung);
				}
				assert(rung >= 0);
				assert(rung < MAX_RUNG);
				dt = 0.5f * rung_dt[rung] * params.t0;
				vx = fmaf(g[XDIM], dt, vx);
				vy = fmaf(g[YDIM], dt, vy);
				vz = fmaf(g[ZDIM], dt, vz);
			}
		}
		shared_reduce_max(max_rung);
		shared_reduce_add(flops);
		if (params.do_phi) {
			shared_reduce_add(pot);
			shared_reduce_add(fx);
			shared_reduce_add(fy);
			shared_reduce_add(fz);
			shared_reduce_add(fnorm);
		}
		if (tid == 0) {
			atomicMax(&params.kreturn->max_rung, max_rung);
			atomicAdd(&params.kreturn->flops, flops);
			if (params.do_phi) {
				atomicAdd(&params.kreturn->pot, pot);
				atomicAdd(&params.kreturn->fx, fy);
				atomicAdd(&params.kreturn->fy, fy);
				atomicAdd(&params.kreturn->fz, fz);
				atomicAdd(&params.kreturn->fnorm, fnorm);
			}
		}
//		atomicAdd(&kicktime, (double) (clock64() - tm));
	} else {
//		tm = clock64();
		const auto children = mycheck.get_children();
		__syncwarp();
		array<fixed32, NDIM> X;
		const int left_active = children[0].get_nactive();
		const int right_active = children[1].get_nactive();
		X[XDIM] = mycheck.get_x(XDIM);
		X[YDIM] = mycheck.get_x(YDIM);
		X[ZDIM] = mycheck.get_x(ZDIM);
		auto& Lnext = (params.lists + bid)->Lexpansion[depth + 1];
		if (left_active && right_active) {
			for (int i = tid; i < EXPANSION_SIZE; i += warpSize) {
				Lnext[i] = Lexpansion[i];
			}
			__syncwarp();
			checklist.push_top();
			do_kick(stack, children[0], depth + 1, X);
			__syncwarp();
			for (int i = tid; i < EXPANSION_SIZE; i += warpSize) {
				Lnext[i] = Lexpansion[i];
			}
			if (tid == 0) {
				checklist.pop_top();
			}
			__syncwarp();
			do_kick(stack, children[1], depth + 1, X);
		} else {
			auto& Lnext = (params.lists + bid)->Lexpansion[depth + 1];
			for (int i = tid; i < EXPANSION_SIZE; i += warpSize) {
				Lnext[i] = Lexpansion[i];
			}
			__syncwarp();
			do_kick(stack, children[left_active ? 0 : 1], depth + 1, X);
			__syncwarp();
		}
		shared_reduce_add(flops);
		if (tid == 0) {
			atomicAdd(&params.kreturn->flops, flops);
		}
	}
//	atomicAdd(&totalflops, flops);

}

__global__ void kick_fmmpm_kernel() {
	const int& tid = threadIdx.x;
	const int& bid = blockIdx.x;
	const auto& params = dev_fmmpm_params;
	if (tid == 0) {
		new (params.lists + bid) list_set();
	}
	__syncwarp();
	auto& checklist = (params.lists + bid)->checklist;
	auto& Lexpansion = (params.lists + bid)->Lexpansion[0];

	const auto inc = [tid,&params]() {
		int j;
		if( tid == 0 ) {
			j = atomicAdd(params.current_index, 1);
		}
		j = __shfl_sync(0xFFFFFFFF,j,0);
		return j;
	};

	for (int j = inc(); j < params.nsink_cells; j = inc()) {
		//PRINT("%i %i\n", j, params.nsink_cells);
		const int cell_index = params.cell_indices[j];
		if (tid == 0) {
			checklist.resize(NCELLS);
		}
		__syncwarp();
		for (int treei = tid; treei < NCELLS; treei += warpSize) {
			checklist[treei].index = params.trees.roots[params.tree_neighbors[cell_index * NCELLS + treei]];
		}
		checkitem mycheck;
		mycheck.index = params.trees.roots[params.tree_neighbors[cell_index * NCELLS + NCELLS / 2]];
		__syncwarp();
		for (int i = tid; i < EXPANSION_SIZE; i += warpSize) {
			Lexpansion[i] = 0.f;
		}
		__syncwarp();
		array<fixed32, NDIM> Lpos;
		Lpos[XDIM] = Lpos[YDIM] = Lpos[ZDIM] = 0.f;
		size_t stack;
		if (mycheck.get_nactive()) {
			do_kick((size_t)(&stack), mycheck, 0, Lpos);
		}
	}
	__syncwarp();
	if (tid == 0) {
		(params.lists + bid)->~list_set();
	}

}

kick_return kick_fmmpm(vector<tree> trees, range<int> box, int min_rung, double scale, double t0, double theta, bool first_call, bool full_eval,
		kick_return* kreturn) {
	bool root;
	if (kreturn == nullptr) {
		kick_return host;
		host.max_rung = -1;
		host.flops = 0;
		host.pot = 0.0;
		host.fnorm = 0.0;
		host.nactive = 0;
		host.fx = 0.0;
		host.fy = 0.0;
		host.fz = 0.0;
		host.pp = 0.0;
		host.pc = 0.0;
		host.cp = 0.0;
		host.cc = 0.0;
		CUDA_CHECK(cudaMalloc(&kreturn, sizeof(kick_return)));
		CUDA_CHECK(cudaMemcpy(kreturn, &host, sizeof(kick_return), cudaMemcpyHostToDevice));
		root = true;
	} else {
		root = false;
	}
	timer tmr;
	tmr.start();
	PRINT("shmem size = %i\n", sizeof(fmmpm_shmem));
	cudaDeviceSetCacheConfig (cudaFuncCachePreferShared);
	timer tm;
	size_t nsources = 0;
	size_t nsinks = 0;
	array<int, NDIM> i;
	const auto bigbox = box.pad(CHAIN_BW);
	const size_t bigvol = bigbox.volume();
	const size_t vol = box.volume();
	int tree_size = 0;
//	PRINT("%i\n", bigvol);
	for (i[0] = bigbox.begin[0]; i[0] != bigbox.end[0]; i[0]++) {
		for (i[1] = bigbox.begin[1]; i[1] != bigbox.end[1]; i[1]++) {
			for (i[2] = bigbox.begin[2]; i[2] != bigbox.end[2]; i[2]++) {
				auto this_cell = chainmesh_get(i);
				nsources += this_cell.pend - this_cell.pbegin;
				if (box.contains(i)) {
					nsinks += this_cell.pend - this_cell.pbegin;
				}
				const int index = bigbox.index(i);
				tree_size += trees[index].size() * sizeof(tree_node) + sizeof(tree);
			}
		}
	}
//	PRINT("tree size = %e GB\n", tree_size / 1024 / 1024 / 1024.0);
	auto phibox = box;
	for (int dim = 0; dim < NDIM; dim++) {
		phibox.begin[dim] *= get_options().four_o_chain;
		phibox.end[dim] *= get_options().four_o_chain;
	}
	phibox = phibox.pad(PHI_BW);
	cudaFuncAttributes attr;
	cudaFuncGetAttributes(&attr, kick_fmmpm_kernel);
	if (attr.maxThreadsPerBlock < WARP_SIZE) {
		PRINT("This CUDA device will not run kick_pme_kernel with the required number of threads (%i)\n", WARP_SIZE);
		abort();
	}
	int occupancy;
	CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor ( &occupancy, kick_fmmpm_kernel,WARP_SIZE, sizeof(fmmpm_shmem)));
	PRINT("Occupancy = %i\n", occupancy);
	int num_blocks = 2 * occupancy * cuda_smp_count();
	//PRINT("%e cells per block\n", (double ) box.volume() / num_blocks);
	const size_t mem_required = mem_requirements(nsources, nsinks, vol, bigvol, phibox.volume()) + tree_size + sizeof(fmmpm_params);
	const size_t free_mem = (size_t) 85 * cuda_free_mem() / size_t(100);
	//PRINT("required = %li freemem = %li\n", mem_required, free_mem);
	size_t value = CUDA_STACK_SIZE;
	CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, value));
	CUDA_CHECK(cudaDeviceGetLimit(&value, cudaLimitStackSize));
	bool fail = false;
	if (value != CUDA_STACK_SIZE) {
		PRINT("Unable to set stack size to %i\n", CUDA_STACK_SIZE);
		fail = true;
	}
	if (fail) {
		abort();
	}
	if (mem_required > free_mem) {
		const auto child_boxes = box.split();
//		PRINT("Splitting\n");
		kick_fmmpm(trees, child_boxes.first, min_rung, scale, t0, theta, first_call, full_eval, kreturn);
		kick_fmmpm(std::move(trees), child_boxes.second, min_rung, scale, t0, theta, first_call, full_eval, kreturn);
	} else {
		cuda_set_device();
		//PRINT("Data transfer\n");
		tm.start();
		fmmpm_params params;
		params.allocate(nsources, nsinks, vol, bigvol, phibox.volume(), num_blocks);
//		tm.stop();
//		PRINT("%e\n", tm.read());
		tm.start();
		params.kreturn = kreturn;
		params.velx = &particles_vel(XDIM, 0);
		params.vely = &particles_vel(YDIM, 0);
		params.velz = &particles_vel(ZDIM, 0);
#ifdef FORCE_TEST
		params.gx = &particles_gforce(XDIM, 0);
		params.gy = &particles_gforce(YDIM, 0);
		params.gz = &particles_gforce(ZDIM, 0);
		params.pot = &particles_pot(0);
#endif
		params.rung = &particles_rung(0);
		params.theta = theta;
		params.min_rung = min_rung;
		params.rs = get_options().rs;
		params.do_phi = full_eval;
		params.rcut = (double) CHAIN_BW / get_options().chain_dim;
		params.rcut2 = sqr(params.rcut);
		params.hsoft = get_options().hsoft;
		params.phi0 = std::pow(get_options().parts_dim, NDIM) * 4.0 * M_PI * sqr(params.rs) - SELF_PHI / params.hsoft;
//		PRINT("RCUT = %e RS\n", params.rcut / params.rs);
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
		int* dev_tree_neighbors = (int*) malloc(sizeof(tree) * NCELLS * vol);
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
		vector<std::pair<size_t, size_t>> cell_sizes;
		int n = 0;
		for (i[0] = bigbox.begin[0]; i[0] != bigbox.end[0]; i[0]++) {
			for (i[1] = bigbox.begin[1]; i[1] != bigbox.end[1]; i[1]++) {
				for (i[2] = bigbox.begin[2]; i[2] != bigbox.end[2]; i[2]++) {
					cell_data entry;
					entry.bigbox_index = bigbox.index(i);
					entry.cell = chainmesh_get(i);
					if (box.contains(i)) {
						const int q = box.index(i);
						cell_sizes.push_back(std::make_pair(n++, trees[bigbox.index(i)].get_nactive(0)));
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
		std::sort(cell_sizes.begin(), cell_sizes.end(), [](std::pair<size_t,size_t> a, std::pair<size_t,size_t> b) {
			return a.second > b.second;
		});
		vector<int> cell_indices;
		for (int j = 0; j < cell_sizes.size(); j++) {
			//	PRINT("%i %i\n", cell_sizes[j].first, cell_sizes[j].second);
			cell_indices.push_back(cell_sizes[j].first);
		}
		CUDA_CHECK(cudaMemcpyAsync(params.cell_indices, cell_indices.data(), sizeof(int) * cell_indices.size(), cudaMemcpyHostToDevice, stream));
		size_t count = 0;
		vector<cpymem> copies;
		for (int j = 0; j < chaincells.size(); j++) {
			auto this_cell = chaincells[j].cell;
			const auto this_size = this_cell.pend - this_cell.pbegin;
			const auto begin = this_cell.pbegin;
			const auto dif = count - begin;
			const int l = chaincells[j].bigbox_index;
			trees[l].adjust_src_indexes(dif);
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

		tree_collection all_trees = tree_collection_create(trees);
		params.trees = all_trees;
		count = 0;
		for (i[0] = box.begin[0]; i[0] != box.end[0]; i[0]++) {
			for (i[1] = box.begin[1]; i[1] != box.end[1]; i[1]++) {
				for (i[2] = box.begin[2]; i[2] != box.end[2]; i[2]++) {
					auto this_cell = chainmesh_get(i);
					const int l = box.index(i);
					array<int, NDIM> j;
					int p = 0;
					for (j[0] = i[0] - CHAIN_BW; j[0] <= i[0] + CHAIN_BW; j[0]++) {
						for (j[1] = i[1] - CHAIN_BW; j[1] <= i[1] + CHAIN_BW; j[1]++) {
							for (j[2] = i[2] - CHAIN_BW; j[2] <= i[2] + CHAIN_BW; j[2]++) {
								const int k = bigbox.index(j);
								dev_tree_neighbors[p + NCELLS * l] = k;
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
//PRINT("Launching kernel\n");
		CUDA_CHECK(cudaMemcpyToSymbol(dev_fmmpm_params, &params, sizeof(fmmpm_params)));
		kick_fmmpm_kernel<<<num_blocks,WARP_SIZE,sizeof(fmmpm_shmem),stream>>>();

		count = 0;
		CUDA_CHECK(cudaStreamSynchronize(stream));
		tm.stop();
//PRINT("%e\n", tm.read());
		tm.reset();
		tm.start();
//PRINT("Transfer back\n");
		copies.resize(0);
		process_copies(std::move(copies), cudaMemcpyDeviceToHost, stream);
		CUDA_CHECK(cudaStreamSynchronize(stream));
		params.free();
		free(dev_tree_neighbors);
		CUDA_CHECK(cudaStreamDestroy(stream));
		tree_collection_destroy(all_trees);
	}
	tmr.stop();
	/*	const double gflops = totalflops / tmr.read() / 1024.0 / 1024.0 / 1024.0;
	 PRINT("Timings\n");
	 double total_time = pctime + cctime + pptime + Ltime + chk1time + chk2time + acttime + longtime + kicktime + branchtime;
	 PRINT("PC  %e\n", pctime / total_time);
	 PRINT("PP  %e\n", pptime / total_time);
	 PRINT("CC  %e\n", cctime / total_time);
	 PRINT("L   %e\n", Ltime / total_time);
	 PRINT("2   %e\n", chk2time / total_time);
	 PRINT("1   %e\n", chk1time / total_time);
	 PRINT("ACT %e\n", acttime / total_time);
	 PRINT("LNG %e\n", longtime / total_time);
	 PRINT("KCK %e\n", kicktime / total_time);
	 PRINT("BRC %e\n", branchtime / total_time);
	 PRINT("GFLOPS = %e\n", gflops);*/

	kick_return host;
	if (root) {
		CUDA_CHECK(cudaMemcpy(&host, kreturn, sizeof(kick_return), cudaMemcpyDeviceToHost));
		CUDA_CHECK(cudaFree(kreturn));
		//PRINT("GFLOPS = %e\n", host.flops / 1024.0 / 1024.0 / 1024.0 / tmr.read());
		tm.stop();
		//PRINT( "%e\n", tm.read());
	}
	return host;
}
