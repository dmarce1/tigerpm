#include <tigerpm/hpx.hpp>
#include <tigerpm/util.hpp>


static range<int> find_my_box(range<int> box, int begin, int end);


range<int> find_my_box(int N) {
	return find_my_box(range<int>(N), 0, hpx_size());
}


static range<int> find_my_box(range<int> box, int begin, int end) {
	if (end - begin == 1) {
		return box;
	} else {
		const int xdim = box.longest_dim();
		auto left = box;
		auto right = box;
		const int mid = (begin + end) / 2;
		const double w = double(mid - begin) / double(end - begin);
		left.end[xdim] = right.begin[xdim] = (((1.0 - w) * box.begin[xdim] + w * box.end[xdim]) + 0.5);
		if (hpx_rank() < mid) {
			assert(hpx_rank() >= begin);
			return find_my_box(left, begin, mid);
		} else {
			assert(hpx_rank() >= mid);
			assert(hpx_rank() < end);
			return find_my_box(right, mid, end);
		}
	}
}
