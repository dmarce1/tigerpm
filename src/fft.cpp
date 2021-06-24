#include <tigerpm/cuda.hpp>
#include <tigerpm/fft.hpp>
#include <tigerpm/hpx.hpp>

#define MAX_BOX_SIZE (8*1024*1024)

static int N;
static std::vector<range<int>> boxes;
static range<int> mybox;
static std::vector<float> Y;

static void find_boxes(range<int> box, int begin, int end, int depth = 0);
static void split_box(range<int> box, std::vector<range<int>>& boxes);

static void transpose(int, int);
static std::vector<float> transpose_swap(const std::vector<float>& out, const range<int>& this_box, int, int);

HPX_PLAIN_ACTION (fft3d_accumulate);
HPX_PLAIN_ACTION (fft3d_init);
HPX_PLAIN_ACTION (fft3d_execute);
HPX_PLAIN_ACTION (fft3d_read);
HPX_PLAIN_ACTION (fft3d_destroy);
HPX_PLAIN_ACTION (transpose_swap);
HPX_PLAIN_ACTION (transpose);

void fft3d_execute() {

	transpose(0, 2);

}

std::vector<float> fft3d_read(const range<int>& this_box) {
	std::vector<hpx::future<void>> futs;
	std::vector<float> data(this_box.volume());
	if (mybox.contains(this_box)) {
		std::array<int, NDIM> i;
		for (i[0] = this_box.begin[0]; i[0] != this_box.end[0]; i[0]++) {
			for (i[1] = this_box.begin[1]; i[1] != this_box.end[1]; i[1]++) {
				for (i[2] = this_box.begin[2]; i[2] != this_box.end[2]; i[2]++) {
					data[this_box.index(i)] = Y[mybox.index(i)];
				}
			}
		}
	} else {
		for (int ri = 0; ri < hpx_size(); ri++) {
			const auto inter = boxes[ri].intersection(this_box);
			if (inter.volume()) {
				std::vector<range<int>> inters;
				split_box(inter, inters);
				for (auto this_inter : inters) {
					auto fut = hpx::async < fft3d_read_action > (hpx_localities()[ri], this_inter);
					futs.push_back(fut.then([this_box,this_inter,&data](hpx::future<std::vector<float>> fut) {
						auto this_data = fut.get();
						std::array<int, NDIM> i;
						for (i[0] = this_inter.begin[0]; i[0] != this_inter.end[0]; i[0]++) {
							for (i[1] = this_inter.begin[1]; i[1] != this_inter.end[1]; i[1]++) {
								for (i[2] = this_inter.begin[2]; i[2] != this_inter.end[2]; i[2]++) {
									data[this_box.index(i)] = this_data[this_inter.index(i)];
								}
							}
						}
					}));
				}
			}
		}
	}
	hpx::wait_all(futs.begin(), futs.end());
	return std::move(data);
}

void fft3d_accumulate(const range<int>& this_box, const std::vector<float>& data) {
	std::vector<hpx::future<void>> futs;
	if (mybox.contains(this_box)) {
		std::array<int, NDIM> i;
		for (i[0] = this_box.begin[0]; i[0] != this_box.end[0]; i[0]++) {
			for (i[1] = this_box.begin[1]; i[1] != this_box.end[1]; i[1]++) {
				for (i[2] = this_box.begin[2]; i[2] != this_box.end[2]; i[2]++) {
					Y[mybox.index(i)] += data[this_box.index(i)];
				}
			}
		}
	} else {
		for (int bi = 0; bi < boxes.size(); bi++) {
			const auto inter = boxes[bi].intersection(this_box);
			if (!inter.empty()) {
				std::vector<range<int>> inters;
				split_box(inter, inters);
				for (auto this_inter : inters) {
					std::vector<float> this_data;
					this_data.resize(this_inter.volume());
					std::array<int, NDIM> i;
					for (i[0] = this_inter.begin[0]; i[0] != this_inter.end[0]; i[0]++) {
						for (i[1] = this_inter.begin[1]; i[1] != this_inter.end[1]; i[1]++) {
							for (i[2] = this_inter.begin[2]; i[2] != this_inter.end[2]; i[2]++) {
								this_data[this_inter.index(i)] = data[this_box.index(i)];
							}
						}
					}
					auto fut = hpx::async < fft3d_accumulate_action
							> (hpx_localities()[bi], this_inter, std::move(this_data));
					futs.push_back(std::move(fut));
				}
			}
		}
	}

	hpx::wait_all(futs.begin(), futs.end());
}

void fft3d_init(int N_) {
	std::vector<hpx::future<void>> futs;
	for (auto c : hpx_children()) {
		futs.push_back(hpx::async < fft3d_init_action > (c, N_));
	}
	N = N_;
	range<int> box;
	for (int dim = 0; dim < NDIM; dim++) {
		box.begin[dim] = 0;
		box.end[dim] = N;
	}
	boxes.resize(hpx_size());
	find_boxes(box, 0, hpx_size());
	mybox = boxes[hpx_rank()];
	Y.resize(mybox.volume(), 0.0);
//	PRINT("Box on rank %i is %s\n", hpx_rank(), mybox.to_string().c_str());
	hpx::wait_all(futs.begin(), futs.end());
}

void fft3d_destroy() {
	std::vector<hpx::future<void>> futs;
	for (auto c : hpx_children()) {
		futs.push_back(hpx::async < fft3d_destroy_action > (c));
	}

	hpx::wait_all(futs.begin(), futs.end());

}

static void find_boxes(range<int> box, int begin, int end, int depth) {
	if (end - begin == 1) {
		boxes[begin] = box;
	} else {
		const int xdim = depth % (NDIM - 1);
		auto left = box;
		auto right = box;
		const int mid = (begin + end) / 2;
		const double w = double(mid - begin) / double(end - begin);
		left.end[xdim] = right.begin[xdim] = (((1.0 - w) * box.begin[xdim] + w * box.end[xdim]) + 0.5);
		find_boxes(left, begin, mid, depth + 1);
		find_boxes(right, mid, end, depth + 1);
	}
}

static std::vector<float> transpose_swap(const std::vector<float>& in, const range<int>& this_box, int dim1, int dim2) {
	std::vector<float> out(this_box.volume());
	auto tbox = this_box.transpose(dim1, dim2);
	assert(mybox.contains(this_box));
	std::array<int, NDIM> i;
	for (i[0] = this_box.begin[0]; i[0] != this_box.end[0]; i[0]++) {
		for (i[1] = this_box.begin[1]; i[1] != this_box.end[1]; i[1]++) {
			for (i[2] = this_box.begin[2]; i[2] != this_box.end[2]; i[2]++) {
				auto ti = i;
				std::swap(ti[dim1], ti[dim2]);
				out[tbox.index(ti)] = Y[mybox.index(i)];
				Y[mybox.index(i)] = in[tbox.index(ti)];
			}
		}
	}
	return out;
}

static void transpose(int dim1, int dim2) {
	std::vector<hpx::future<void>> futs;
	for (auto c : hpx_children()) {
		futs.push_back(hpx::async < transpose_action > (c, dim1, dim2));
	}
	range<int> tbox = mybox.transpose(dim1, dim2);
	for (int bi = hpx_rank(); bi < boxes.size(); bi++) {
		const auto tinter = boxes[bi].intersection(tbox);
		//	printf("%i with %i Intersection : %s\n", hpx_rank(), bi, tinter.to_string().c_str());
		std::vector<float> data;
		if (!tinter.empty()) {
			std::vector<range<int>> tinters;
			split_box(tinter, tinters);
			for (auto this_tinter : tinters) {
				const auto inter = this_tinter.transpose(dim1, dim2);
				data.resize(inter.volume());
				std::array<int, NDIM> i;
				for (i[0] = inter.begin[0]; i[0] != inter.end[0]; i[0]++) {
					for (i[1] = inter.begin[1]; i[1] != inter.end[1]; i[1]++) {
						for (i[2] = inter.begin[2]; i[2] != inter.end[2]; i[2]++) {
							const int k = inter.index(i);
							const int l = mybox.index(i);
							assert(k < data.size());
							assert(l < Y.size());
							data[k] = Y[l];
						}
					}
				}
				auto fut = hpx::async < transpose_swap_action
						> (hpx_localities()[bi], std::move(data), this_tinter, dim1, dim2);
				futs.push_back(fut.then([inter](hpx::future<std::vector<float>> fut) {
					auto data = fut.get();
					std::array<int, NDIM> i;
					for (i[0] = inter.begin[0]; i[0] != inter.end[0]; i[0]++) {
						for (i[1] = inter.begin[1]; i[1] != inter.end[1]; i[1]++) {
							for (i[2] = inter.begin[2]; i[2] != inter.end[2]; i[2]++) {
								Y[mybox.index(i)] = data[inter.index(i)];
							}
						}
					}
				}));
			}
		}
	}

	hpx::wait_all(futs.begin(), futs.end());

}

static void split_box(range<int> box, std::vector<range<int>>& boxes) {
	if (box.volume() < MAX_BOX_SIZE) {
		boxes.push_back(box);
	} else {
		auto children = box.split();
		split_box(children.first, boxes);
		split_box(children.second, boxes);
	}
}
