#include <tigerpm/gravity_long.hpp>
#include <tigerpm/fixed.hpp>
#include <tigerpm/options.hpp>
#include <tigerpm/particles.hpp>
#include <tigerpm/util.hpp>

__global__ void compute_source_kernel(float* source, range<int> source_box, fixed32* x, fixed32* y, fixed32* z, int nparts, float N) {
	const int& tid = threadIdx.x;
	const int& bid = blockIdx.x;
	const int& bsz = blockDim.x;
	const int& gsz = gridDim.x;
	const int begin = size_t(bid) * size_t(nparts) / size_t(gsz);
	const int end = size_t(bid + 1) * size_t(nparts) / size_t(gsz);
	for (int i = begin + tid; i < end; i += bsz) {
		array<double, NDIM> X;
		array<int, NDIM> I;
		array<array<double, CLOUD_W>, NDIM> w;
		X[XDIM] = x[i].to_float();
		X[YDIM] = y[i].to_float();
		X[ZDIM] = z[i].to_float();
		for (int dim = 0; dim < NDIM; dim++) {
			I[dim] = int(X[dim] * N + PHI_BW - 1) - PHI_BW;
			for (int i = 0; i < CLOUD_W; i++) {
				w[dim][i] = cloud4(X[dim] * N - I[dim] - i);
			}
		}
		array<int, NDIM> J;
		const float c0 = 4.0 * M_PI * N;
		for (J[0] = 0; J[0] < CLOUD_W; J[0]++) {
			for (J[1] = 0; J[1] < CLOUD_W; J[1]++) {
				for (J[2] = 0; J[2] < CLOUD_W; J[2]++) {
					const int i = source_box.index(I[0] + J[0], I[1] + J[1], I[2] + J[2]);
					const float value = c0 * w[0][J[0]] * w[1][J[1]] * w[2][J[2]];
					atomicAdd(source + i, value);
				}
			}
		}
	}

}

static vector<float> phi;
static range<int> source_box;

void compute_source_cuda(int pbegin, int pend, float* dev_src, cudaStream_t stream, float N) {
	size_t sz = sizeof(float) * source_box.volume();
	sz += NDIM * sizeof(fixed32) * (pend - pbegin);
	if (sz > cuda_free_mem() * 85 / 100) {
		const int mid = (pbegin + pend) / 2;
		compute_source_cuda(pbegin, mid, dev_src, stream, N);
		CUDA_CHECK(cudaStreamSynchronize(stream));
		compute_source_cuda(mid, pend, dev_src, stream, N);
		CUDA_CHECK(cudaStreamSynchronize(stream));
	} else {
		fixed32* x;
		fixed32* y;
		fixed32* z;
		const int count = (pend - pbegin);
		CUDA_CHECK(cudaMallocAsync(&x, sizeof(fixed32) * count, stream));
		CUDA_CHECK(cudaMallocAsync(&y, sizeof(fixed32) * count, stream));
		CUDA_CHECK(cudaMallocAsync(&z, sizeof(fixed32) * count, stream));
		CUDA_CHECK(cudaMemcpyAsync(x, &particles_pos(XDIM,pbegin), sizeof(fixed32)* count, cudaMemcpyHostToDevice, stream));
		CUDA_CHECK(cudaMemcpyAsync(y, &particles_pos(YDIM,pbegin), sizeof(fixed32)* count, cudaMemcpyHostToDevice, stream));
		CUDA_CHECK(cudaMemcpyAsync(z, &particles_pos(ZDIM,pbegin), sizeof(fixed32)* count, cudaMemcpyHostToDevice, stream));
		int occupancy;
		cudaFuncAttributes attr;
		CUDA_CHECK(cudaFuncGetAttributes(&attr, compute_source_kernel));
		CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, compute_source_kernel, attr.numRegs, 0));
		int num_blocks = cuda_smp_count() * occupancy;
		compute_source_kernel<<<num_blocks,attr.numRegs, 0, stream>>>(dev_src, source_box, x, y, z, count, N);
		CUDA_CHECK(cudaFreeAsync(x, stream));
		CUDA_CHECK(cudaFreeAsync(y, stream));
		CUDA_CHECK(cudaFreeAsync(z, stream));
	}
}

vector<float> gravity_long_compute_source_local() {

	source_box = find_my_box(get_options().chain_dim);
	for (int dim = 0; dim < NDIM; dim++) {
		const static auto ratio = get_options().four_o_chain;
		source_box.begin[dim] *= ratio;
		source_box.end[dim] *= ratio;
	}
	source_box = source_box.pad(PHI_BW);
	const int vol = source_box.volume();
	vector<float> source;
	source.resize(vol, 0.0f);
	const float N = get_options().four_dim;
	source_box = find_my_box(get_options().chain_dim);
	for (int dim = 0; dim < NDIM; dim++) {
		const static auto ratio = get_options().four_o_chain;
		source_box.begin[dim] *= ratio;
		source_box.end[dim] *= ratio;
	}
	source_box = source_box.pad(PHI_BW);
	source.resize(source_box.volume(), 0.0f);

	float* dev_source;
	cudaStream_t stream;
	CUDA_CHECK(cudaStreamCreate(&stream));
	CUDA_CHECK(cudaMallocAsync(&dev_source, vol * sizeof(float), stream));
	CUDA_CHECK(cudaMemcpyAsync(dev_source, source.data(), vol * sizeof(float), cudaMemcpyHostToDevice, stream));
	compute_source_cuda(0, particles_size(), dev_source, stream, N);
	CUDA_CHECK(cudaMemcpyAsync(source.data(), dev_source, vol * sizeof(float), cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK(cudaFreeAsync(dev_source, stream));
	CUDA_CHECK(cudaStreamSynchronize(stream));
	CUDA_CHECK(cudaStreamDestroy(stream));
	return source;
}

