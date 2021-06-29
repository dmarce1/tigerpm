#include <tigerpm/chainmesh.hpp>
#include <tigerpm/particles.hpp>

std::vector<chaincell> cells;

void chainmesh_create() {
	const auto box = particles_get_local_box();
	box.pad(CHAIN_BW);
	cells.resize(box.volume());
	const auto counts = particles_mesh_count();
	for (int i = 0; i < counts.size(); i++) {
		cells[i].particles.reserve(counts[i]);
	}
	const int nthreads = hpx::thread::hardware_concurrency();
	std::vector<hpx::future<void>> futs;
	for (int proc = 0; proc < nthreads; proc++) {
		const auto func = [proc,nthreads,box]() {
			const int begin = size_t(proc) * size_t(particles_size()) / size_t(nthreads);
			const int end = size_t(proc+1) * size_t(particles_size()) / size_t(nthreads);
			for (int part_index = begin; part_index < end; part_index++) {
				const auto loc = particles_mesh_loc(part_index);
				const int chain_index = box.index(loc);
				auto& cell = cells[chain_index];
				while( (*cell.lock)++ != 0 ) {
					(*cell.lock)--;
				}
				cell.particles.push_back(part_index);
				(*cell.lock)--;
			}
		};
		futs.push_back(hpx::async(func));
	}
	hpx::wait_all(futs.begin(), futs.end());
}
