#include <tigerpm/tree.hpp>
#include <tigerpm/hpx.hpp>
#include <tigerpm/kick_treepm.hpp>
#include <tigerpm/particles.hpp>
#include <tigerpm/timer.hpp>

HPX_PLAIN_ACTION(kick_treepm_begin);
HPX_PLAIN_ACTION(kick_treepm_end);

void kick_treepm_begin(int min_rung, double scale, double t0, bool first_call) {
	vector<hpx::future<void>> futs;
	for (auto c : hpx_children()) {
		futs.push_back(hpx::async<kick_treepm_begin_action>(c, min_rung, scale, t0, first_call));
	}

	vector<tree> trees;
	vector<vector<sink_bucket>> buckets;
	const auto box = chainmesh_interior_box();
	const auto vol = box.volume();
	timer tm;
	tm.start();
	array<int, NDIM> i;
	std::vector<hpx::future<void>> futs1;
	trees.resize(vol);
	buckets.resize(vol);
	for (i[0] = box.begin[0]; i[0] < box.end[0]; i[0]++) {
		for (i[1] = box.begin[1]; i[1] < box.end[1]; i[1]++) {
			for (i[2] = box.begin[2]; i[2] < box.end[2]; i[2]++) {
				const auto func = [i,box,&trees,&buckets]() {
					const auto cell = chainmesh_get(i);
					const auto rc = tree_create(i,cell);
					const auto index = box.index(i);
					trees[index] = std::move(rc.first);
					buckets[index] = std::move(rc.second);
				};
				futs1.push_back(hpx::async(func));
			}
		}
	}
	hpx::wait_all(futs1.begin(), futs1.end());
	tm.stop();
	PRINT("Trees took %e s\n", tm.read());
	tm.reset();

	kick_treepm(trees, buckets, box, min_rung, scale, t0, first_call);

	hpx::wait_all(futs.begin(), futs.end());

}

void kick_treepm_end() {
	vector<hpx::future<void>> futs;
	for (auto c : hpx_children()) {
		futs.push_back(hpx::async<kick_treepm_end_action>(c));
	}

	particles_resize(particles_local_size());



	hpx::wait_all(futs.begin(), futs.end());

}
