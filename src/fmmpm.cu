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

#define FMMPM_MIN_THREADS 16
#define FMMPM_BLOCK_SIZE 32

#include <algorithm>

__constant__ float rung_dt[MAX_RUNG] = { 1.0 / (1 << 0), 1.0 / (1 << 1), 1.0 / (1 << 2), 1.0 / (1 << 3), 1.0 / (1 << 4), 1.0 / (1 << 5), 1.0 / (1 << 6), 1.0
		/ (1 << 7), 1.0 / (1 << 8), 1.0 / (1 << 9), 1.0 / (1 << 10), 1.0 / (1 << 11), 1.0 / (1 << 12), 1.0 / (1 << 13), 1.0 / (1 << 14), 1.0 / (1 << 15), 1.0
		/ (1 << 16), 1.0 / (1 << 17), 1.0 / (1 << 18), 1.0 / (1 << 19), 1.0 / (1 << 20), 1.0 / (1 << 21), 1.0 / (1 << 22), 1.0 / (1 << 23), 1.0 / (1 << 24), 1.0
		/ (1 << 25), 1.0 / (1 << 26), 1.0 / (1 << 27), 1.0 / (1 << 28), 1.0 / (1 << 29), 1.0 / (1 << 30), 1.0 / (1 << 31) };

#define LIST_SIZE  (8*1024)
#define STACK_SIZE (32*1024)
#define MAX_DEPTH 64

struct checkitem {
	int index;
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
	int get_src_begin() const {
		return tr->get_pbegin(index);
	}
	CUDA_EXPORT inline
	int get_src_end() const {
		return tr->get_pend(index);
	}
	CUDA_EXPORT inline
	int get_snk_begin() const {
		return tr->get_snk_begin(index);
	}
	CUDA_EXPORT inline
	int get_snk_end() const {
		return tr->get_snk_end(index);
	}
	CUDA_EXPORT inline
	multipole get_multipole() const {
		return tr->get_multipole(index);
	}
	CUDA_EXPORT inline
	array<checkitem, 2> get_children() {
		const auto indices = tr->get_children(index);
		array<checkitem, 2> c;
		c[0].index = indices[0];
		c[1].index = indices[1];
		c[0].tr = c[1].tr = tr;
		return c;
	}
};

struct list_set {
	stack_vector<checkitem, STACK_SIZE, MAX_DEPTH> checklist;
	fixedcapvec<checkitem, LIST_SIZE> openlist;
	fixedcapvec<checkitem, LIST_SIZE> nextlist;
	fixedcapvec<checkitem, LIST_SIZE> multilist;
	fixedcapvec<checkitem, LIST_SIZE> cplist;
	fixedcapvec<checkitem, LIST_SIZE> pplist;
	fixedcapvec<checkitem, LIST_SIZE> tmplist;
	array<expansion, MAX_DEPTH> Lexpansion;
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
		CUDA_CHECK(cudaMalloc(&lists, nblocks * sizeof(list_set)));
		CUDA_CHECK(cudaMalloc(&rung, sink_size * sizeof(char)));
		CUDA_CHECK(cudaMalloc(&phi, phi_cell_count * sizeof(float)));
		CUDA_CHECK(cudaMalloc(&active, nblocks * sizeof(int) * FMM_BUCKET_SIZE));
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
	array<array<float, NDIM>, FMM_BUCKET_SIZE> g;
	array<float, FMM_BUCKET_SIZE> phi;
	array<fixed32, FMM_BUCKET_SIZE> x;
	array<fixed32, FMM_BUCKET_SIZE> y;
	array<fixed32, FMM_BUCKET_SIZE> z;
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
__device__ inline void shared_reduce_add(T& number) {
	for (int P = warpSize / 2; P >= 1; P /= 2) {
		number += __shfl_xor_sync(0xffffffff, number, P);
	}
}

template<class T>
__device__ inline void shared_reduce_min(T& number) {
	for (int P = warpSize / 2; P >= 1; P /= 2) {
		number = fminf(number, __shfl_xor_sync(0xffffffff, number, P));
	}
}

inline __device__ void compute_pp_interaction(float dx, float dy, float dz, float& gx, float& gy, float& gz, float& phi) {
	const fmmpm_params& params = dev_fmmpm_params;
	const float& twooversqrtpi = params.twooversqrtpi;
	const float& h2 = params.h2;
	const float& hinv = params.hinv;
	const float& h2inv = params.h2inv;
	const float& h3inv = params.h3inv;
	const float& inv2rs = params.inv2rs;
	const float rcut2 = sqr(params.rcut);
	const float r2 = sqr(dx, dy, dz);
	float rinv, rinv3;
	if (r2 < rcut2) {
		float exp0;
		rinv = r2 > 0.0f ? rsqrtf(r2) : 0.0f;
		const float r = r2 * rinv;
		const float r0 = r * inv2rs;
		const float erfc0 = erfcexp(r0, &exp0);
		if (r2 > h2) {
			rinv3 = (erfc0 + twooversqrtpi * r0 * exp0) * rinv * rinv * rinv;
			rinv *= erfc0;
		} else {
			const float q2 = r2 * h2inv;
			float d1 = +15.0f / 8.0f;
			d1 = fmaf(d1, q2, -21.0f / 4.0f);
			d1 = fmaf(d1, q2, +35.0f / 8.0f);
			d1 *= h3inv;
			rinv3 = ((erfc0 - 1.0f) + twooversqrtpi * r0 * exp0) * rinv * rinv * rinv + d1;
			if (params.do_phi) {
				float d0 = -5.0f / 16.0f;
				d0 = fmaf(d0, q2, 21.0f / 16.0f);
				d0 = fmaf(d0, q2, -35.0f / 16.0f);
				d0 = fmaf(d0, q2, 35.0f / 16.0f);
				d0 *= hinv;
				rinv = (erfc0 - 1.0f) * rinv + d0;
			}
		}
		gx -= dx * rinv3;
		gy -= dy * rinv3;
		gz -= dz * rinv3;
		phi -= rinv;
	}
}

__device__ void pp_interactions(int nactive) {
	const int& tid = threadIdx.x;
	const int& bid = blockIdx.x;
	__shared__ extern int shmem_ptr[];
	fmmpm_shmem& shmem = (fmmpm_shmem&) (*shmem_ptr);
	const fmmpm_params& params = dev_fmmpm_params;
	const auto& list = (params.lists + bid)->pplist;
	if( list.size() == 0 ) {
		PRINT( "ZERO\n");
		return;
	}
	int i = 0;
	int N = 0;
	int part_index;
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
				compute_pp_interaction(dx, dy, dz, g[XDIM], g[YDIM], g[ZDIM], phi);
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
				compute_pp_interaction(dx, dy, dz, g[XDIM], g[YDIM], g[ZDIM], phi);
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
		}
	}

}

__device__ void cc_interactions(checkitem mycheck, expansion& Lexpansion) {
	const int& tid = threadIdx.x;
	const int& bid = blockIdx.x;
	__shared__ extern int shmem_ptr[];
	const fmmpm_params& params = dev_fmmpm_params;
	const auto& list = (params.lists + bid)->multilist;

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
		expansion D;
		for (int j = 0; j < EXPANSION_SIZE; j++) {
			D[j] = 0.f;
		}
		greens_function(D, dx, dy, dz, params.inv2rs);
		const auto M = list[i].get_multipole();
		M2L_kernel(L, M, D, params.do_phi);
	}
	for (int i = 0; i < EXPANSION_SIZE; i++) {
		shared_reduce_add(L[i]);
	}
	if (tid == 0) {
		Lexpansion = L;
	}
	__syncwarp();
}

__device__ void long_range_interp(int nactive) {
	const int& tid = threadIdx.x;
	__shared__ extern int shmem_ptr[];
	fmmpm_shmem& shmem = (fmmpm_shmem&) (*shmem_ptr);
	const auto& params = dev_fmmpm_params;
	for (int sink_index = tid; sink_index < nactive; sink_index += warpSize) {
		array<float, NDIM>& g = shmem.g[sink_index];
		float& phi = shmem.phi[sink_index];
		g[XDIM] = g[YDIM] = g[ZDIM] = 0.f;
		phi = params.phi0;
		const fixed32 sink_x = shmem.x[sink_index];
		const fixed32 sink_y = shmem.y[sink_index];
		const fixed32 sink_z = shmem.z[sink_index];
		array<int, NDIM> I;
		array<int, NDIM> J;
		array<float, NDIM> X;
		X[XDIM] = sink_x.to_float();
		X[YDIM] = sink_y.to_float();
		X[ZDIM] = sink_z.to_float();
		array<array<float, NINTERP>, NDIM> w;
		array<array<float, NINTERP>, NDIM> dw;
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
		for (J[0] = I[0]; J[0] < I[0] + NINTERP; J[0]++) {
			for (J[1] = I[1]; J[1] < I[1] + NINTERP; J[1]++) {
				for (J[2] = I[2]; J[2] < I[2] + NINTERP; J[2]++) {
					const int l = params.phi_box.index(J);
					const int i0 = J[0] - I[0];
					const int i1 = J[1] - I[1];
					const int i2 = J[2] - I[2];
					const float phi0 = params.phi[l];
					const float phi1 = phi0 * params.Nfour;
					double w0 = dw[0][i0] * w[1][i1] * w[2][i2];
					g[XDIM] -= w0 * phi1;
					w0 = w[0][i0] * dw[1][i1] * w[2][i2];
					g[YDIM] -= w0 * phi1;
					w0 = w[0][i0] * w[1][i1] * dw[2][i2];
					g[ZDIM] -= w0 * phi1;
					if (params.do_phi) {
						w0 = w[0][i0] * w[1][i1] * w[2][i2];
						phi += w0 * phi0;
					}
				}
			}
		}
	}
	__syncwarp();
}

__device__ void do_kick(checkitem mycheck, int depth, array<fixed32, NDIM> Lpos) {
	if (depth >= MAX_DEPTH) {
		PRINT("MAX_DEPTH exceeded!\n");
		__trap();
	}
	const int& tid = threadIdx.x;
	const int& bid = blockIdx.x;
	__shared__ extern int shmem_ptr[];
	fmmpm_shmem& shmem = (fmmpm_shmem&) (*shmem_ptr);
	const auto& params = dev_fmmpm_params;
	const auto& theta2inv = 1.0f / sqr(params.theta);
	auto* lists = params.lists + bid;
	auto& checklist = lists->checklist;
	auto& multilist = lists->multilist;
	auto& pplist = lists->pplist;
//	auto& cplist = lists->cplist;
//	auto& tmplist =lists->tmplist;
	auto& openlist = lists->openlist;
	auto& nextlist = lists->nextlist;
	auto& Lexpansion = lists->Lexpansion[depth];
	auto* active = params.active + bid * FMM_BUCKET_SIZE;
	const bool iamleaf = mycheck.is_leaf();
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
	if (tid == 0) {
		Lexpansion = Ltmp;
		multilist.resize(0);
		pplist.resize(0);
	}
	__syncwarp();
	do {
		__syncwarp();
		if (tid == 0) {
			nextlist.resize(0);
			openlist.resize(0);
		}
		__syncwarp();
		const int maxi = round_up(checklist.size(), warpSize);
		bool multi;
		bool part;
		bool open;
		bool next;
		for (int ci = tid; ci < maxi; ci += warpSize) {
			checkitem check;
			if (ci < checklist.size()) {
				check = checklist[ci];
				const bool source_isleaf = check.is_leaf();
				const fixed32 src_x = check.get_x(XDIM);
				const fixed32 src_y = check.get_x(YDIM);
				const fixed32 src_z = check.get_x(ZDIM);
				const float source_radius = check.get_radius();
				const float dx = distance(sink_x, src_x);
				const float dy = distance(sink_y, src_y);
				const float dz = distance(sink_z, src_z);
				const float R2 = sqr(dx, dy, dz);
				const bool far = R2 > sqr(myradius + source_radius) * theta2inv;
				multi = far;
				part = !far && (source_isleaf && iamleaf);
				open = !far && (source_isleaf && !iamleaf);
				next = !far && !source_isleaf;
			} else {
				multi = false;
				part = false;
				open = false;
				next = false;
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

			index = part;
			index = compute_indices(index, total) + pplist.size();
			__syncwarp();
			if (tid == 0) {
				pplist.resize(pplist.size() + total);
			}
			__syncwarp();
			if (part) {
				pplist[index] = check;
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

			index = open;
			index = compute_indices(index, total) + openlist.size();
			__syncwarp();
			if (tid == 0) {
				openlist.resize(openlist.size() + total);
			}
			__syncwarp();
			if (open) {
				openlist[index] = check;
			}
		}

		__syncwarp();
		if (tid == 0) {
			checklist.resize(nextlist.size() * 2);
		}
		__syncwarp();
		for (int ci = tid; ci < nextlist.size(); ci += warpSize) {
			assert(!nextlist[ci].is_leaf());
			const auto children = nextlist[ci].get_children();
			for (int i = 0; i < 2; i++) {
				checklist[2 * ci + i] = children[i];
			}
		}
		const int offset = checklist.size();
		__syncwarp();
		if (tid == 0) {
			checklist.resize(offset + openlist.size());
		}
		__syncwarp();
		for (int ci = tid; ci < openlist.size(); ci += warpSize) {
			checklist[ci + offset] = openlist[ci];
		}
	} while (iamleaf && checklist.size());
//	if( nextlist.size() > 1024 || openlist.size() > 1024 || multilist.size() > 1024 || pplist.size() > 1024)
	if( iamleaf) {
//		PRINT( "%i %i %i %i %i \n", checklist.size(), nextlist.size(), openlist.size(), multilist.size(), pplist.size());
	}

	cc_interactions(mycheck, Lexpansion);

	if (iamleaf) {
		const auto snk_begin = mycheck.get_snk_begin();
		const auto snk_end = mycheck.get_snk_end();
		const int nsinks = snk_end - snk_begin;
		const int imax = round_up(nsinks, warpSize);
		int nactive = 0;
		for (int i = tid; i < imax; i += warpSize) {
			const int snki = snk_begin + i;
			bool is_active;
			if (i < nsinks) {
				is_active = (params.rung[snki] >= params.min_rung);
			} else {
				is_active = false;
			}
			int total;
			int active_index = compute_indices(int(is_active), total) + nactive;
			int srci = mycheck.get_src_begin() + i;
			if (is_active) {
				active[active_index] = snki;
			}
			nactive += total;
			shmem.x[active_index] = params.x[srci];
			shmem.y[active_index] = params.y[srci];
			shmem.z[active_index] = params.z[srci];
			__syncwarp();
		}

		long_range_interp(nactive);

		pp_interactions(nactive);

		for (int sink_index = tid; sink_index < nactive; sink_index += warpSize) {
			array<float, NDIM>& g = shmem.g[sink_index];
			float& phi = shmem.phi[sink_index];
			dX[XDIM] = distance(shmem.x[sink_index], Lpos[XDIM]);
			dX[YDIM] = distance(shmem.y[sink_index], Lpos[YDIM]);
			dX[ZDIM] = distance(shmem.z[sink_index], Lpos[ZDIM]);
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
		}

	} else {
		const auto children = mycheck.get_children();
		__syncwarp();
		if (tid == 0) {
			(params.lists + bid)->Lexpansion[depth + 1] = Lexpansion;
		}
		__syncwarp();
		array<fixed32, NDIM> X;
		X[XDIM] = mycheck.get_x(XDIM);
		X[YDIM] = mycheck.get_x(YDIM);
		X[ZDIM] = mycheck.get_x(ZDIM);
		checklist.push_top();
		do_kick(children[0], depth + 1, X);
		__syncwarp();
		if (tid == 0) {
			(params.lists + bid)->Lexpansion[depth + 1] = Lexpansion;
			checklist.pop_top();
		}
		__syncwarp();
		do_kick(children[1], depth + 1, X);
	}

}

__global__ void kick_fmmpm_kernel() {
	const int& tid = threadIdx.x;
	const int& bid = blockIdx.x;
	const int& gsz = gridDim.x;
	const auto& params = dev_fmmpm_params;
	if (tid == 0) {
		new (params.lists + bid) list_set();
	}
	__syncwarp();
	auto& checklist = (params.lists + bid)->checklist;
	auto& Lexpansion = (params.lists + bid)->Lexpansion[0];
	const int cell_begin = size_t(bid) * (size_t) params.nsink_cells / (size_t) gsz;
	const int cell_end = size_t(bid + 1) * (size_t) params.nsink_cells / (size_t) gsz;
	for (int cell_index = cell_begin; cell_index < cell_end; cell_index++) {
		if (tid == 0) {
			checklist.resize(NCELLS);
		}
		__syncwarp();
		for (int treei = tid; treei < NCELLS; treei += warpSize) {
			checklist[treei].tr = params.tree_neighbors + cell_index * NCELLS + treei;
			checklist[treei].index = 0;
		}
		checkitem mycheck;
		mycheck.tr = params.tree_neighbors + cell_index * NCELLS + NCELLS / 2;
		mycheck.index = 0;
		__syncwarp();
		for (int i = tid; i < EXPANSION_SIZE; i += warpSize) {
			Lexpansion[i] = 0.f;
		}
		__syncwarp();
		array<fixed32, NDIM> Lpos;
		Lpos[XDIM] = Lpos[YDIM] = Lpos[ZDIM] = 0.f;
		do_kick(mycheck, 0, Lpos);
	}
	__syncwarp();
	if (tid == 0) {
		(params.lists + bid)->~list_set();
	}

}

#define STACK_SIZE (32*1024)

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
				if (box.contains(i)) {
					nsinks += this_cell.pend - this_cell.pbegin;
				}
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
	size_t value = STACK_SIZE;
	CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, value));
	CUDA_CHECK(cudaDeviceGetLimit(&value, cudaLimitStackSize));
	bool fail = false;
	if (value != STACK_SIZE) {
		PRINT("Unable to set stack size to %i\n", STACK_SIZE);
		fail = true;
	}
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
		params.theta = 0.5;
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
		for (i[0] = box.begin[0]; i[0] != box.end[0]; i[0]++) {
			for (i[1] = box.begin[1]; i[1] != box.end[1]; i[1]++) {
				for (i[2] = box.begin[2]; i[2] != box.end[2]; i[2]++) {
					auto this_cell = chainmesh_get(i);
					const auto this_size = this_cell.pend - this_cell.pbegin;
					const auto begin = this_cell.pbegin;
					cpymem cpy;
					const int l = box.index(i);
					const int m = bigbox.index(i);
					const auto dif = count - begin;
					trees[m].adjust_snk_indexes(dif);
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
					count += this_size;
				}
			}
		}
		count = 0;
		vector<tree> dev_trees(bigvol);
		for (int j = 0; j < bigvol; j++) {
			dev_trees[j] = trees[j].to_device();
			dev_trees[j].nodes = dev_all_trees + count;
			std::memcpy(all_trees.data() + count, trees[j].nodes, sizeof(tree_node) * trees[j].size());
			count += dev_trees[j].size();
		}
		CUDA_CHECK(cudaMemcpyAsync(dev_all_trees, all_trees.data(), trees_size * sizeof(tree_node), cudaMemcpyHostToDevice));
		for (i[0] = box.begin[0]; i[0] != box.end[0]; i[0]++) {
			for (i[1] = box.begin[1]; i[1] != box.end[1]; i[1]++) {
				for (i[2] = box.begin[2]; i[2] != box.end[2]; i[2]++) {
					auto this_cell = chainmesh_get(i);
					const int l = box.index(i);
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
