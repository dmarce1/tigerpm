#include <tigerpm/tree.hpp>
#include <tigerpm/hpx.hpp>
#include <tigerpm/fmmpm.hpp>
#include <tigerpm/particles.hpp>
#include <tigerpm/timer.hpp>

HPX_PLAIN_ACTION(kick_fmmpm_begin);
HPX_PLAIN_ACTION(kick_fmmpm_end);

void kick_fmmpm_begin(int min_rung, double scale, double t0, bool first_call) {
	vector<hpx::future<void>> futs;
	for (auto c : hpx_children()) {
		futs.push_back(hpx::async<kick_fmmpm_begin_action>(c, min_rung, scale, t0, first_call));
	}

	vector<tree> trees;
	const auto box = chainmesh_interior_box();
	const auto bigbox = box.pad(CHAIN_BW);
	const auto vol = box.volume();
	const auto bigvol = bigbox.volume();
		timer tm;
	tm.start();
	array<int, NDIM> i;
	std::vector<hpx::future<void>> futs1;
	trees.resize(bigvol);
	for (i[0] = box.begin[0]; i[0] < box.end[0]; i[0]++) {
		for (i[1] = box.begin[1]; i[1] < box.end[1]; i[1]++) {
			for (i[2] = box.begin[2]; i[2] < box.end[2]; i[2]++) {
				const auto func = [i,bigbox,box,&trees]() {
					const auto cell = chainmesh_get(i);
					const auto rc = tree_create(i,cell);
					const auto index = bigbox.index(i);
					trees[index] = std::move(rc.first);
				};
				futs1.push_back(hpx::async(func));
			}
		}
	}
	const int N = get_options().chain_dim;
	hpx::wait_all(futs1.begin(), futs1.end());
	tm.stop();
	for (i[0] = bigbox.begin[0]; i[0] < bigbox.end[0]; i[0]++) {
		for (i[1] = bigbox.begin[1]; i[1] < bigbox.end[1]; i[1]++) {
			for (i[2] = bigbox.begin[2]; i[2] < bigbox.end[2]; i[2]++) {
				if( !box.contains(i)) {
					auto j = i;
					for( int dim = 0; dim < NDIM; dim++) {
						if( j[dim] < 0 ) {
							j[dim] += N;
						} else if( j[dim] >= N) {
							j[dim] -= N;
						}
					}
					trees[bigbox.index(i)] = trees[bigbox.index(j)];
				}
			}
		}
	}
	PRINT("Trees took %e s\n", tm.read());
	tm.reset();

	kick_fmmpm(trees, box, min_rung, scale, t0, first_call);

	hpx::wait_all(futs.begin(), futs.end());

}

void kick_fmmpm_end() {
	vector<hpx::future<void>> futs;
	for (auto c : hpx_children()) {
		futs.push_back(hpx::async<kick_fmmpm_end_action>(c));
	}

	particles_resize(particles_local_size());



	hpx::wait_all(futs.begin(), futs.end());

}
