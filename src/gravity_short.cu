#include <tigerpm/cuda.hpp>
#include <tigerpm/gravity_short.hpp>
#include <tigerpm/particles.hpp>
#include <tigerpm/util.hpp>

__global__ void gravity_ewald_kernel(fixed32* sinkx, fixed32* sinky, fixed32* sinkz, fixed32* sourcex, fixed32* sourcey,
		fixed32* sourcez, int Nsource, double* rphi, double* rgx, double* rgy, double*rgz);

std::pair<std::vector<double>, std::array<std::vector<double>, NDIM>> gravity_short_ewald_call_kernel(
		const std::vector<fixed32>& sinkx, const std::vector<fixed32>& sinky, const std::vector<fixed32>& sinkz) {
	std::pair<std::vector<double>, std::array<std::vector<double>, NDIM>> rc;
	fixed32* dev_sinkx;
	fixed32* dev_sinky;
	fixed32* dev_sinkz;
	fixed32* dev_srcx;
	fixed32* dev_srcy;
	fixed32* dev_srcz;
	double* dev_phi;
	double* dev_gx;
	double* dev_gy;
	double* dev_gz;
	const int Nsinks = sinkx.size();
	CUDA_CHECK(cudaMalloc(&dev_sinkx, Nsinks * sizeof(fixed32)));
	CUDA_CHECK(cudaMalloc(&dev_sinky, Nsinks * sizeof(fixed32)));
	CUDA_CHECK(cudaMalloc(&dev_sinkz, Nsinks * sizeof(fixed32)));
	CUDA_CHECK(cudaMalloc(&dev_phi, Nsinks * sizeof(double)));
	CUDA_CHECK(cudaMalloc(&dev_gx, Nsinks * sizeof(double)));
	CUDA_CHECK(cudaMalloc(&dev_gy, Nsinks * sizeof(double)));
	CUDA_CHECK(cudaMalloc(&dev_gz, Nsinks * sizeof(double)));
	std::vector<double> zero(Nsinks,0.0);
	CUDA_CHECK(cudaMemcpy(dev_phi, zero.data(), Nsinks*sizeof(double),cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(dev_gx, zero.data(), Nsinks*sizeof(double),cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(dev_gy, zero.data(), Nsinks*sizeof(double),cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(dev_gz, zero.data(), Nsinks*sizeof(double),cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(dev_sinkx, sinkx.data(), Nsinks * sizeof(fixed32), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(dev_sinky, sinky.data(), Nsinks * sizeof(fixed32), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(dev_sinkz, sinkz.data(), Nsinks * sizeof(fixed32), cudaMemcpyHostToDevice));
	PRINT( "%li free\n", cuda_free_mem());
	const int parts_per_loop = cuda_free_mem() / (NDIM * sizeof(fixed32)) * 85 / 100;
	int occupancy;
	CUDA_CHECK(
			cudaOccupancyMaxActiveBlocksPerMultiprocessor ( &occupancy, gravity_ewald_kernel,EWALD_BLOCK_SIZE, sizeof(double)*(NDIM+1)*EWALD_BLOCK_SIZE ));
	int num_kernels = std::max((int)(occupancy * cuda_smp_count() / Nsinks),1);
	std::vector < cudaStream_t > streams(num_kernels);
	for (int i = 0; i < num_kernels; i++) {
		cudaStreamCreate (&streams[i]);
	}
	PRINT( "%i particles per loop, %i kernels\n", parts_per_loop, num_kernels);
	for (int i = 0; i < particles_size(); i += parts_per_loop) {
		const int total_size = std::min(size_t(particles_size()), size_t(i) + size_t(parts_per_loop)) - size_t(i);
		CUDA_CHECK(cudaMalloc(&dev_srcx, total_size * sizeof(fixed32)));
		CUDA_CHECK(cudaMalloc(&dev_srcy, total_size * sizeof(fixed32)));
		CUDA_CHECK(cudaMalloc(&dev_srcz, total_size * sizeof(fixed32)));
		CUDA_CHECK(cudaMemcpy(dev_srcx, &particles_pos(0, i), total_size * sizeof(fixed32), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(dev_srcy, &particles_pos(1, i), total_size * sizeof(fixed32), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(dev_srcz, &particles_pos(2, i), total_size * sizeof(fixed32), cudaMemcpyHostToDevice));
		for (int j = 0; j < num_kernels; j++) {
			const int begin = size_t(j) * size_t(total_size) / size_t(num_kernels);
			const int end = size_t(j + 1) * size_t(total_size) / size_t(num_kernels);
			gravity_ewald_kernel<<<Nsinks,EWALD_BLOCK_SIZE,0,streams[i]>>>(dev_sinkx, dev_sinky, dev_sinkz, dev_srcx + begin, dev_srcy + begin,
					dev_srcz + begin, end - begin, dev_phi, dev_gx, dev_gz, dev_gz);
		}
		for (int i = 0; i < num_kernels; i++) {
			cudaStreamSynchronize (streams[i]);
		}
		CUDA_CHECK(cudaFree(dev_srcx));
		CUDA_CHECK(cudaFree(dev_srcy));
		CUDA_CHECK(cudaFree(dev_srcz));
	}
	for (int i = 0; i < num_kernels; i++) {
		cudaStreamDestroy (streams[i]);
	}
	rc.first.resize(Nsinks);
	for (int dim = 0; dim < NDIM; dim++) {
		rc.second[dim].resize(Nsinks);
	}
	CUDA_CHECK(cudaMemcpy(rc.first.data(), dev_phi, sizeof(double) * Nsinks, cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(rc.second[XDIM].data(),dev_gx,sizeof(double)*Nsinks,cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(rc.second[YDIM].data(),dev_gy,sizeof(double)*Nsinks,cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(rc.second[ZDIM].data(),dev_gz,sizeof(double)*Nsinks,cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaFree(dev_phi));
	CUDA_CHECK(cudaFree(dev_gx));
	CUDA_CHECK(cudaFree(dev_gy));
	CUDA_CHECK(cudaFree(dev_gz));
	CUDA_CHECK(cudaFree(dev_sinkx));
	CUDA_CHECK(cudaFree(dev_sinky));
	CUDA_CHECK(cudaFree(dev_sinkz));
	return rc;
}

__global__ void gravity_ewald_kernel(fixed32* sinkx, fixed32* sinky, fixed32* sinkz, fixed32* sourcex, fixed32* sourcey,
		fixed32* sourcez, int Nsource, double* rphi, double* rgx, double* rgy, double*rgz) {

	__shared__ double phi[EWALD_BLOCK_SIZE];
	__shared__ double gx[EWALD_BLOCK_SIZE];
	__shared__ double gy[EWALD_BLOCK_SIZE];
	__shared__ double gz[EWALD_BLOCK_SIZE];

	const int& tid = threadIdx.x;
	const int& bid = blockIdx.x;

	const fixed32 x = sinkx[bid];
	const fixed32 y = sinky[bid];
	const fixed32 z = sinkz[bid];

	phi[tid] = gx[tid] = gy[tid] = gz[tid] = 0.0;
	for (int sourcei = tid; sourcei < Nsource; sourcei += EWALD_BLOCK_SIZE) {
		const float X = distance(x, sourcex[sourcei]);
		const float Y = distance(y, sourcey[sourcei]);
		const float Z = distance(z, sourcez[sourcei]);
		if (sourcei < Nsource) {
			for (int xi = -4; xi <= +4; xi++) {
				for (int yi = -4; yi <= +4; yi++) {
					for (int zi = -4; zi < +4; zi++) {
						const float dx = X - xi;
						const float dy = Y - yi;
						const float dz = Z - zi;
						const float r2 = sqr(dx, dy, dz);
						if (r2 > 0.f && r2 < 2.6f * 2.6f) {
							const float r = sqrt(r2);
							const float rinv = 1.f / r;
							const float r2inv = rinv * rinv;
							const float r3inv = r2inv * rinv;
							const float exp0 = expf(-4.f * r2);
							const float erfc0 = erfcf(2.f * r);
							const float expfactor = 4.0 / sqrt(M_PI) * r * exp0;
							const float d0 = -erfc0 * rinv;
							const float d1 = (expfactor + erfc0) * r3inv;
							phi[tid] += d0;
							for (int dim = 0; dim < NDIM; dim++) {
								gx[tid] -= dx * d1;
								gy[tid] -= dy * d1;
								gz[tid] -= dz * d1;
							}
						}
					}
				}
			}
			phi[tid] += float(M_PI / 4.f);
			for (int xi = -3; xi <= +3; xi++) {
				for (int yi = -3; yi <= +3; yi++) {
					for (int zi = -3; zi < +3; zi++) {
						const float hx = xi;
						const float hy = yi;
						const float hz = zi;
						const float h2 = sqr(hx, hy, hz);
						if (h2 > 0.0f && h2 <= 8) {
							const float hdotx = X * hx + Y * hy + Z * hz;
							const float omega = 2.0f * M_PI * hdotx;
							const float c = cosf(omega);
							const float s = sinf(omega);
							const float c0 = -1.0f / h2 * expf(float(-M_PI * M_PI * 0.25f) * h2) * float(1.f / M_PI);
							const float c1 = -s * 2.0 * M_PI * c0;
							phi[tid] += c0 * c;
							gx[tid] += c1 * hx;
							gy[tid] += c1 * hy;
							gz[tid] += c1 * hz;
						}
					}
				}
			}
		}
	}
	__syncthreads();
	for (int P = EWALD_BLOCK_SIZE / 2; P >= 1; P /= 2) {
		if (tid < P) {
			gx[tid] += gx[tid + P];
			gy[tid] += gy[tid + P];
			gz[tid] += gz[tid + P];
			phi[tid] += phi[tid + P];
		}
		__syncthreads();
	}
	if (tid == 0) {
		atomicAdd(rphi + bid, phi[0]);
		atomicAdd(rgx + bid, gx[0]);
		atomicAdd(rgy + bid, gy[0]);
		atomicAdd(rgz + bid, gz[0]);
	}

}
