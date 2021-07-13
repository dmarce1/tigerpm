#include <tigerpm/hpx.hpp>
#include <tigerpm/drift.hpp>
#include <tigerpm/particles.hpp>
#include <tigerpm/util.hpp>

void drift(double scale, double dt) {
	std::vector<hpx::future<void>> futs;
	const int nthreads = hpx::thread::hardware_concurrency();
	for (int proc = 0; proc < nthreads; proc++) {
		const auto func = [=]() {
			const double factor = dt / scale;
			const int begin = size_t(proc) * size_t(particles_size()) / size_t(nthreads);
			const int end = size_t(proc+1) * size_t(particles_size()) / size_t(nthreads);
			for( int i = begin; i < end; i++) {
				double x = particles_pos(XDIM,i).to_double();
				double y = particles_pos(YDIM,i).to_double();
				double z = particles_pos(ZDIM,i).to_double();
				const float vx = particles_vel(XDIM,i);
				const float vy = particles_vel(YDIM,i);
				const float vz = particles_vel(ZDIM,i);
				x += double(vx) * factor;
				y += double(vy) * factor;
				z += double(vz) * factor;
				constrain_range(x);
				constrain_range(y);
				constrain_range(z);
				particles_pos(XDIM,i) = x;
				particles_pos(YDIM,i) = y;
				particles_pos(ZDIM,i) = z;
			}
		};
		futs.push_back(hpx::async(func));
	}
	hpx::wait_all(futs.begin(), futs.end());

}
