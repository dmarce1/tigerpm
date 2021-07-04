#include <tigerpm/kick_treepm.hpp>
#include <tigerpm/timer.hpp>
#include <tigerpm/options.hpp>
#include <tigerpm/particles.hpp>
#include <tigerpm/util.hpp>
#include <tigerpm/gravity_long.hpp>
#include <tigerpm/gravity_short.hpp>

#include <thrust/device_vector.h>

#include <algorithm>

struct fixed4 {
	fixed32 x;
	fixed32 y;
	fixed32 z;
	float m;
};

__constant__ float rung_dt[MAX_RUNG] = { 1.0 / (1 << 0), 1.0 / (1 << 1), 1.0 / (1 << 2), 1.0 / (1 << 3), 1.0 / (1 << 4), 1.0 / (1 << 5), 1.0 / (1 << 6), 1.0
		/ (1 << 7), 1.0 / (1 << 8), 1.0 / (1 << 9), 1.0 / (1 << 10), 1.0 / (1 << 11), 1.0 / (1 << 12), 1.0 / (1 << 13), 1.0 / (1 << 14), 1.0 / (1 << 15), 1.0
		/ (1 << 16), 1.0 / (1 << 17), 1.0 / (1 << 18), 1.0 / (1 << 19), 1.0 / (1 << 20), 1.0 / (1 << 21), 1.0 / (1 << 22), 1.0 / (1 << 23), 1.0 / (1 << 24), 1.0
		/ (1 << 25), 1.0 / (1 << 26), 1.0 / (1 << 27), 1.0 / (1 << 28), 1.0 / (1 << 29), 1.0 / (1 << 30), 1.0 / (1 << 31) };

struct sink_cell {
	int begin;
	int end;
	array<int, NDIM> loc;
};

#define WORKSPACE_SIZE  1024
#define INTERSPACE_SIZE (2 * TREEPM_BLOCK_SIZE * SOURCE_BUCKET_SIZE)

struct treepm_params {
	fixed32* x;
	fixed32* y;
	fixed32* z;
	float* velx;
	float* vely;
	float* velz;
	float* phi;
	char* rung;
	int* checklist;
	int* nextlist;
	fixed4* sourcelist;
	tree* tree_neighbors;
	sink_bucket** buckets;
	int* bucket_cnt;
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
	void allocate(size_t source_size, size_t sink_size, size_t cell_count, size_t big_cell_count, size_t phi_cell_count, int nblocks) {
		nsink_cells = cell_count;
		CUDA_CHECK(cudaMalloc(&tree_neighbors, cell_count * NCELLS * sizeof(tree)));
		CUDA_CHECK(cudaMalloc(&buckets, cell_count * sizeof(sink_bucket*)));
		CUDA_CHECK(cudaMalloc(&bucket_cnt, cell_count * sizeof(int)));
		CUDA_CHECK(cudaMalloc(&x, source_size * sizeof(fixed32)));
		CUDA_CHECK(cudaMalloc(&y, source_size * sizeof(fixed32)));
		CUDA_CHECK(cudaMalloc(&z, source_size * sizeof(fixed32)));
		CUDA_CHECK(cudaMalloc(&velx, sink_size * sizeof(float)));
		CUDA_CHECK(cudaMalloc(&vely, sink_size * sizeof(float)));
		CUDA_CHECK(cudaMalloc(&velz, sink_size * sizeof(float)));
		CUDA_CHECK(cudaMalloc(&rung, sink_size * sizeof(char)));
		CUDA_CHECK(cudaMalloc(&phi, phi_cell_count * sizeof(float)));
		CUDA_CHECK(cudaMalloc(&checklist, nblocks * sizeof(int) * WORKSPACE_SIZE));
		CUDA_CHECK(cudaMalloc(&nextlist, nblocks * sizeof(int) * WORKSPACE_SIZE));
		CUDA_CHECK(cudaMalloc(&sourcelist, nblocks * sizeof(fixed4) * INTERSPACE_SIZE));
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
		CUDA_CHECK(cudaFree(buckets));
		CUDA_CHECK(cudaFree(bucket_cnt));
		CUDA_CHECK(cudaFree(phi));
		CUDA_CHECK(cudaFree(rung));
		CUDA_CHECK(cudaFree(tree_neighbors));
		CUDA_CHECK(cudaFree(checklist));
		CUDA_CHECK(cudaFree(nextlist));
		CUDA_CHECK(cudaFree(sourcelist));
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
	mem += vol * sizeof(sink_cell);
	mem += 2 * sizeof(int) * vol;
	mem += sizeof(sink_bucket*) * vol;
	mem += sizeof(int) * vol;
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

struct treepm_shmem {
	array<array<float, NDIM>, SINK_BUCKET_SIZE> g;
	array<float, SINK_BUCKET_SIZE> phi;
	array<int, SINK_BUCKET_SIZE> active_srci;
	array<int, SINK_BUCKET_SIZE> active_snki;
	array<float, TREEPM_BLOCK_SIZE> reduce;
	array<int, TREEPM_BLOCK_SIZE> index;
	array<int, TREEPM_BLOCK_SIZE> partlist;
};

__constant__ treepm_params dev_treepm_params;

__device__ int compute_indices(array<int, TREEPM_BLOCK_SIZE>& index) {
	const int& tid = threadIdx.x;
	for (int P = 1; P < TREEPM_BLOCK_SIZE; P *= 2) {
		int tmp;
		__syncthreads();
		if (tid >= P) {
			tmp = index[tid - P];
		}
		__syncthreads();
		if (tid >= P) {
			index[tid] += tmp;
		}
	}
	__syncthreads();
	return (tid > 0 ? index[tid - 1] : 0);

}

__global__ void kick_treepm_kernel() {
	const treepm_params& params = dev_treepm_params;
	__shared__ treepm_shmem shmem;
	const int& tid = threadIdx.x;
	const int& bid = blockIdx.x;
	const int& gsz = gridDim.x;
	const float& inv2rs = params.inv2rs;
	const float& twooversqrtpi = params.twooversqrtpi;
	const float& h2 = params.h2;
	const float& hinv = params.hinv;
	const float& h3inv = params.h3inv;
	const float rcut = 5.f * params.rs;
	const float theta2inv = 1.0f / sqr(params.theta);
	const int cell_begin = size_t(bid) * (size_t) params.nsink_cells / (size_t) gsz;
	const int cell_end = size_t(bid + 1) * (size_t) params.nsink_cells / (size_t) gsz;
	for (int cell_index = cell_begin; cell_index < cell_end; cell_index++) {
		sink_bucket* buckets = params.buckets[cell_index];
		const int& bucket_cnt = params.bucket_cnt[cell_index];
		for (int bi = 0; bi < bucket_cnt; bi++) {
			const auto& bucket = buckets[bi];
			const auto& snk_begin = bucket.snk_begin;
			const auto& snk_end = bucket.snk_end;
			__syncthreads();
			const int nsinks = snk_end - snk_begin;
			const int imax = round_up(nsinks, TREEPM_BLOCK_SIZE);
			int nactive = 0;
			for (int i = tid; i < imax; i += TREEPM_BLOCK_SIZE) {
				const int this_index = snk_begin + i;
				bool is_active;
				if (i < nsinks) {
					is_active = int(params.rung[this_index] >= params.min_rung);
				} else {
					is_active = false;
				}
				shmem.index[tid] = int(is_active);
				int active_index = compute_indices(shmem.index) + +nactive;
				if (is_active) {
					shmem.active_snki[active_index] = this_index;
					shmem.active_srci[active_index] = bucket.src_begin + i;
				}
				nactive += shmem.index[TREEPM_BLOCK_SIZE - 1];
				__syncthreads();
			}
			for (int sink_index = tid; sink_index < nactive; sink_index += TREEPM_BLOCK_SIZE) {
				array<float, NDIM>& g = shmem.g[sink_index];
				float& phi = shmem.phi[sink_index];
				g[0] = g[1] = g[2] = 0.f;
				phi = 0.f;
				const int srci = shmem.active_srci[sink_index];
				const fixed32 sink_x = params.x[srci];
				const fixed32 sink_y = params.y[srci];
				const fixed32 sink_z = params.z[srci];
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
			const size_t offset = bid * WORKSPACE_SIZE;
			int* checklist = params.checklist + offset;
			int* nextlist = params.nextlist + offset;
			fixed4* sourcelist = params.sourcelist + bid * INTERSPACE_SIZE;
			const auto& sink_x = bucket.x[XDIM];
			const auto& sink_y = bucket.x[YDIM];
			const auto& sink_z = bucket.x[ZDIM];
			const auto& sink_radius = bucket.radius;
			for (int tree_index = 0; tree_index < NCELLS; tree_index++) {
				tree& tr = params.tree_neighbors[cell_index * NCELLS + tree_index];
				int check_size = 1;
				int next_size = 0;
				int source_size = 0;
				checklist[0] = 0;
				const auto process_sources = [&]() {
					for (int sink_index = tid; sink_index < nactive; sink_index += TREEPM_BLOCK_SIZE) {
						float& phi = shmem.phi[sink_index];
						auto& g = shmem.g[sink_index];
						const int srci = shmem.active_srci[sink_index];
						const fixed32 sink_x = params.x[srci];
						const fixed32 sink_y = params.y[srci];
						const fixed32 sink_z = params.z[srci];
						for (int i = 0; i < source_size; i++) {
							const fixed32& src_x = sourcelist[i].x;
							const fixed32& src_y = sourcelist[i].y;
							const fixed32& src_z = sourcelist[i].z;
							const float& m = sourcelist[i].m;
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
							rinv3 *= m;
							rinv *= m;
							g[XDIM] -= dx * rinv3;
							g[YDIM] -= dy * rinv3;
							g[ZDIM] -= dz * rinv3;
							phi -= rinv;
						}
					}
				};
				while (check_size) {
					const int maxi = round_up(check_size, TREEPM_BLOCK_SIZE);
					for (int ci = tid; ci < maxi; ci += TREEPM_BLOCK_SIZE) {
						int this_index = 0;
						bool nextb = false;
						bool multib = false;
						bool partb = false;
						fixed32 source_x;
						fixed32 source_y;
						fixed32 source_z;
						int index;
						if (ci < check_size) {
							index = checklist[ci];
							source_x = tr.get_x(0, index);
							source_y = tr.get_x(1, index);
							source_z = tr.get_x(2, index);
							const int nparts = tr.get_pend(index) - tr.get_pbegin(index);
							const float source_radius = tr.get_radius(index);
							const float dx = distance(sink_x, source_x);
							const float dy = distance(sink_y, source_y);
							const float dz = distance(sink_z, source_z);
							const float R2 = sqr(dx, dy, dz);
							const float R = sqrtf(R2);
							const bool cutoff = R - source_radius - sink_radius > rcut;
							const bool far = R2 > sqr(source_radius + sink_radius) * theta2inv;
							//		PRINT( "%e %e %e\n", R2, source_radius, sink_radius);
							const bool leaf = nparts <= SOURCE_BUCKET_SIZE;
							multib = !cutoff && far;
							partb = !cutoff && !far && leaf;
							nextb = !cutoff && !far && !leaf;
							//		PRINT( "%i %i %i \n", multib, partb, nextb);
						}
						shmem.index[tid] = multib;
						this_index = compute_indices(shmem.index) + source_size;
						if (source_size + shmem.index[TREEPM_BLOCK_SIZE - 1] >= INTERSPACE_SIZE) {
							PRINT("internal interspace exceeded on multipoles\n");
							__trap();
							assert(false);
						}
						if (multib) {
							sourcelist[this_index].x = source_x;
							sourcelist[this_index].y = source_y;
							sourcelist[this_index].z = source_z;
							sourcelist[this_index].m = tr.get_mass(index);
						}
						source_size += shmem.index[TREEPM_BLOCK_SIZE - 1];
						__syncthreads();

						shmem.index[tid] = partb;
						this_index = compute_indices(shmem.index);
						if (partb) {
							shmem.partlist[this_index] = index;
						}
						const int part_size = shmem.index[TREEPM_BLOCK_SIZE - 1];
						__syncthreads();

						for (int j = 0; j < part_size; j++) {
							const int begin = tr.get_pbegin(shmem.partlist[j]);
							const int end = tr.get_pend(shmem.partlist[j]);
							for (int k = begin + tid; k < end; k += TREEPM_BLOCK_SIZE) {
								const int l = source_size + k - begin;
								sourcelist[l].x = params.x[k];
								sourcelist[l].y = params.y[k];
								sourcelist[l].z = params.z[k];
								sourcelist[l].m = 1.f;
							}
							source_size += end - begin;
						}

						shmem.index[tid] = nextb;
						this_index = compute_indices(shmem.index);
						if (next_size + shmem.index[TREEPM_BLOCK_SIZE - 1] >= WORKSPACE_SIZE) {
							PRINT("internal workspace exceeded\n");
							__trap();
							assert(false);
						}
						if (nextb) {
							const auto children = tr.get_children(index);
							nextlist[next_size + 2 * this_index + 0] = children[0];
							nextlist[next_size + 2 * this_index + 1] = children[1];
						}
						next_size += 2 * shmem.index[TREEPM_BLOCK_SIZE - 1];
						__syncthreads();

					}
					auto tmp1 = nextlist;
					nextlist = checklist;
					checklist = tmp1;
					check_size = next_size;
					next_size = 0;
					if (source_size > INTERSPACE_SIZE / 2) {
						process_sources();
						source_size = 0;
					}
				}
				process_sources();
				for (int sink_index = tid; sink_index < nactive; sink_index += TREEPM_BLOCK_SIZE) {
					array<float, NDIM>& g = shmem.g[sink_index];
					float& phi = shmem.phi[sink_index];
					const int snki = shmem.active_snki[sink_index];
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
}

void kick_treepm(vector<tree> trees, vector<vector<sink_bucket>> buckets, range<int> box, int min_rung, double scale, double t0, bool first_call) {
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
	cudaFuncAttributes attr;
	cudaFuncGetAttributes(&attr, kick_treepm_kernel);
	if (attr.maxThreadsPerBlock < TREEPM_BLOCK_SIZE) {
		PRINT("This CUDA device will not run kick_pme_kernel with the required number of threads (%i)\n", TREEPM_BLOCK_SIZE);
		abort();
	}
	int occupancy;
	CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor ( &occupancy, kick_treepm_kernel,TREEPM_BLOCK_SIZE, sizeof(treepm_shmem)));
	int num_blocks = TREEPM_OVERSUBSCRIPTION * occupancy * cuda_smp_count();
	const size_t mem_required = mem_requirements(nsources, nsinks, vol, bigvol, phibox.volume()) + tree_size + buckets_size
			+ num_blocks * sizeof(int) * 2 * (WORKSPACE_SIZE + INTERSPACE_SIZE);
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
		params.allocate(nsources, nsinks, vol, bigvol, phibox.volume(), num_blocks);
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
			if (chaincells[j].box_index >= 0) {
				const int q = chaincells[j].box_index;
				for (auto& bucket : buckets[q]) {
					bucket.src_begin += dif;
					bucket.src_end += dif;
				}
			}
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
		vector<tree> dev_trees(bigvol);
		for (int j = 0; j < bigvol; j++) {
			dev_trees[j] = trees[j].to_device(stream);
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
					for (auto& bucket : buckets[l]) {
						bucket.snk_begin += dif;
						bucket.snk_end += dif;
					}
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
		vector<sink_bucket*> dev_buckets;
		vector<int> bucket_count;
		timer tm1;
		tm1.start();
		for (int j = 0; j < vol; j++) {
			bucket_count.push_back(buckets[j].size());
			sink_bucket* bucket;
			CUDA_CHECK(cudaMalloc(&bucket, sizeof(sink_bucket) * buckets[j].size()));
			CUDA_CHECK(cudaMemcpyAsync(bucket, buckets[j].data(), sizeof(sink_bucket) * buckets[j].size(), cudaMemcpyHostToDevice, stream));
			dev_buckets.push_back(bucket);
		}
		tm1.stop();
		PRINT("bucket time = %e\n", tm1.read());
		CUDA_CHECK(cudaMemcpyAsync(params.bucket_cnt, bucket_count.data(), sizeof(int) * vol, cudaMemcpyHostToDevice, stream));
		CUDA_CHECK(cudaMemcpyAsync(params.buckets, dev_buckets.data(), sizeof(sink_bucket*) * vol, cudaMemcpyHostToDevice, stream));
		CUDA_CHECK(cudaMemcpyAsync(params.tree_neighbors, dev_tree_neighbors, sizeof(tree) * NCELLS * vol, cudaMemcpyHostToDevice, stream));
		process_copies(std::move(copies), cudaMemcpyHostToDevice, stream);
		CUDA_CHECK(cudaStreamSynchronize(stream));
		tm.stop();
		PRINT("%e\n", tm.read());
		tm.reset();
		tm.start();
		PRINT("Launching kernel\n");
		CUDA_CHECK(cudaMemcpyToSymbol(dev_treepm_params, &params, sizeof(treepm_params)));
		kick_treepm_kernel<<<num_blocks,TREEPM_BLOCK_SIZE,0,stream>>>();

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
		tm.stop();
		for (int j = 0; j < vol; j++) {
			CUDA_CHECK(cudaFree(dev_buckets[j]));
		}

		PRINT("%e\n", tm.read());
	}
}
