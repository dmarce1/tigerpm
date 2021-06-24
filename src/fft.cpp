#include <tigerpm/cuda.hpp>
#include <tigerpm/fft.hpp>
#include <tigerpm/hpx.hpp>

#include <fftw3.h>

#define MAX_BOX_SIZE (8*1024*1024)

static int N;
static std::vector<range<int>> real_boxes;
static std::array<std::vector<range<int>>, NDIM> cmplx_boxes;
static range<int> real_mybox;
static std::array<range<int>, NDIM> cmplx_mybox;

static std::vector<float> R;
static std::vector<cmplx> Y;
static std::vector<cmplx> Y1;

static void find_boxes(range<int> box, std::vector<range<int>>& boxes, int begin, int end);
static void split_box(range<int> box, std::vector<range<int>>& real_boxes);
static void update();
static void transpose(int, int);

void fft3d_phase1();
void fft3d_phase2(int);
static std::vector<cmplx> transpose_read(const range<int>&, int dim1, int dim2);

HPX_PLAIN_ACTION (fft3d_accumulate);
HPX_PLAIN_ACTION (fft3d_init);
HPX_PLAIN_ACTION (fft3d_execute);
HPX_PLAIN_ACTION (fft3d_phase1);
HPX_PLAIN_ACTION (fft3d_phase2);
HPX_PLAIN_ACTION (fft3d_read_real);
HPX_PLAIN_ACTION (fft3d_read_complex);
HPX_PLAIN_ACTION (transpose_read);
HPX_PLAIN_ACTION (fft3d_destroy);
HPX_PLAIN_ACTION (transpose);
HPX_PLAIN_ACTION (update);

#define XDIM 0
#define YDIM 1
#define ZDIM 2

void fft3d_execute() {
	fft3d_phase1();
	transpose(1, 2);
	fft3d_phase2(1);
	transpose(2, 1);
	update();
	transpose(0, 2);
	fft3d_phase2(0);
	transpose(2, 0);
	update();
}

void update() {
	std::vector<hpx::future<void>> futs;
	for (auto c : hpx_children()) {
		futs.push_back(hpx::async < update_action > (c));
	}
	Y = Y1;
	hpx::wait_all(futs.begin(), futs.end());
}

void fft3d_phase1() {
	std::vector<hpx::future<void>> futs;
	for (auto c : hpx_children()) {
		futs.push_back(hpx::async < fft3d_phase1_action > (c));
	}
	std::array<int, NDIM> i;
	Y.resize(cmplx_mybox[ZDIM].volume());
	for (i[0] = real_mybox.begin[0]; i[0] != real_mybox.end[0]; i[0]++) {
		for (i[1] = real_mybox.begin[1]; i[1] != real_mybox.end[1]; i[1]++) {
			fftw_complex out[N / 2 + 1];
			double in[N];
			fftw_plan p;
			for (i[2] = 0; i[2] != N; i[2]++) {
				in[i[2]] = R[real_mybox.index(i)];
			}
			p = fftw_plan_dft_r2c_1d(N, in, out, 0);
			fftw_execute(p);
			fftw_destroy_plan(p);
			for (i[2] = 0; i[2] < N / 2 + 1; i[2]++) {
				const int l = cmplx_mybox[ZDIM].index(i);
				Y[l].real() = out[i[2]][0];
				Y[l].imag() = out[i[2]][1];
			}
		}
	}
	hpx::wait_all(futs.begin(), futs.end());
}

void fft3d_phase2(int dim) {
	std::vector<hpx::future<void>> futs;
	for (auto c : hpx_children()) {
		futs.push_back(hpx::async < fft3d_phase2_action > (c, dim));
	}
	Y = std::move(Y1);
	std::array<int, NDIM> i;
	for (i[0] = cmplx_mybox[dim].begin[0]; i[0] != cmplx_mybox[dim].end[0]; i[0]++) {
		for (i[1] = cmplx_mybox[dim].begin[1]; i[1] != cmplx_mybox[dim].end[1]; i[1]++) {
			fftw_complex out[N];
			fftw_complex in[N];
			fftw_plan p;
			for (i[2] = 0; i[2] < N; i[2]++) {
				const auto l = cmplx_mybox[dim].index(i);
				in[i[2]][0] = Y[l].real();
				in[i[2]][1] = Y[l].imag();
			}
			p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
			fftw_execute(p);
			fftw_destroy_plan(p);
			for (i[2] = 0; i[2] < N; i[2]++) {
				const int l = cmplx_mybox[dim].index(i);
				Y[l].real() = out[i[2]][0];
				Y[l].imag() = out[i[2]][1];
			}
		}
	}
	hpx::wait_all(futs.begin(), futs.end());
}

std::vector<float> fft3d_read_real(const range<int>& this_box) {
	std::vector<hpx::future<void>> futs;
	std::vector<float> data(this_box.volume());
	if (real_mybox.contains(this_box)) {
		std::array<int, NDIM> i;
		for (i[0] = this_box.begin[0]; i[0] != this_box.end[0]; i[0]++) {
			for (i[1] = this_box.begin[1]; i[1] != this_box.end[1]; i[1]++) {
				for (i[2] = this_box.begin[2]; i[2] != this_box.end[2]; i[2]++) {
					data[this_box.index(i)] = R[real_mybox.index(i)];
				}
			}
		}
	} else {
		for (int ri = 0; ri < hpx_size(); ri++) {
			const auto inter = real_boxes[ri].intersection(this_box);
			if (inter.volume()) {
				std::vector<range<int>> inters;
				split_box(inter, inters);
				for (auto this_inter : inters) {
					auto fut = hpx::async < fft3d_read_real_action > (hpx_localities()[ri], this_inter);
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

std::vector<cmplx> fft3d_read_complex(const range<int>& this_box) {
	std::vector<hpx::future<void>> futs;
	std::vector<cmplx> data(this_box.volume());
	const auto mybox = cmplx_mybox[ZDIM];
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
			const auto inter = real_boxes[ri].intersection(this_box);
			if (inter.volume()) {
				std::vector<range<int>> inters;
				split_box(inter, inters);
				for (auto this_inter : inters) {
					auto fut = hpx::async < fft3d_read_complex_action > (hpx_localities()[ri], this_inter);
					futs.push_back(fut.then([this_box,this_inter,&data](hpx::future<std::vector<cmplx>> fut) {
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
	if (real_mybox.contains(this_box)) {
		std::array<int, NDIM> i;
		for (i[0] = this_box.begin[0]; i[0] != this_box.end[0]; i[0]++) {
			for (i[1] = this_box.begin[1]; i[1] != this_box.end[1]; i[1]++) {
				for (i[2] = this_box.begin[2]; i[2] != this_box.end[2]; i[2]++) {
					R[real_mybox.index(i)] += data[this_box.index(i)];
				}
			}
		}
	} else {
		for (int bi = 0; bi < real_boxes.size(); bi++) {
			const auto inter = real_boxes[bi].intersection(this_box);
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
	real_boxes.resize(hpx_size());
	find_boxes(box, real_boxes, 0, hpx_size());
	real_mybox = real_boxes[hpx_rank()];
	R.resize(real_mybox.volume(), 0.0);
	for (int dim = 0; dim < NDIM; dim++) {
		for (int dim1 = 0; dim1 < NDIM; dim1++) {
			box.begin[dim1] = 0;
			box.end[dim1] = N;
		}
		box.end[dim] = N / 2 + 1;
		cmplx_boxes[dim].resize(hpx_size());
		find_boxes(box, cmplx_boxes[dim], 0, hpx_size());
		cmplx_mybox[dim] = cmplx_boxes[dim][hpx_rank()];
	}
	hpx::wait_all(futs.begin(), futs.end());
}

void fft3d_destroy() {
	std::vector<hpx::future<void>> futs;
	for (auto c : hpx_children()) {
		futs.push_back(hpx::async < fft3d_destroy_action > (c));
	}

	hpx::wait_all(futs.begin(), futs.end());

}

static void find_boxes(range<int> box, std::vector<range<int>>& boxes, int begin, int end) {
	if (end - begin == 1) {
		boxes[begin] = box;
	} else {
		const int xdim = box.longest_dim();
		auto left = box;
		auto right = box;
		const int mid = (begin + end) / 2;
		const double w = double(mid - begin) / double(end - begin);
		left.end[xdim] = right.begin[xdim] = (((1.0 - w) * box.begin[xdim] + w * box.end[xdim]) + 0.5);
		find_boxes(left, boxes, begin, mid);
		find_boxes(right, boxes, mid, end);
	}
}

static void transpose(int dim1, int dim2) {
	std::vector<hpx::future<void>> futs;
	for (auto c : hpx_children()) {
		futs.push_back(hpx::async < transpose_action > (c, dim1, dim2));
	}
	range<int> tbox = cmplx_mybox[dim1].transpose(dim1, dim2);
	Y1.resize(cmplx_mybox[dim1].volume());
	for (int bi = 0; bi < cmplx_boxes[dim2].size(); bi++) {
		const auto tinter = cmplx_boxes[dim2][bi].intersection(tbox);
		std::vector<float> data;
		if (!tinter.empty()) {
			std::vector<range<int>> tinters;
			split_box(tinter, tinters);
			for (auto this_tinter : tinters) {
				auto inter = this_tinter.transpose(dim1, dim2);
				auto fut = hpx::async < transpose_read_action > (hpx_localities()[bi], this_tinter, dim1, dim2);
				futs.push_back(fut.then([inter,dim1](hpx::future<std::vector<cmplx>> fut) {
					auto data = fut.get();
					std::array<int, NDIM> i;
					for (i[0] = inter.begin[0]; i[0] != inter.end[0]; i[0]++) {
						for (i[1] = inter.begin[1]; i[1] != inter.end[1]; i[1]++) {
							for (i[2] = inter.begin[2]; i[2] != inter.end[2]; i[2]++) {
								const int k = inter.index(i);
								const int l = cmplx_mybox[dim1].index(i);
								assert(k < data.size());
								assert(l < Y.size());
								Y1[l] = data[k];
							}
						}
					}
				}));
			}
		}
	}

	hpx::wait_all(futs.begin(), futs.end());

}

static std::vector<cmplx> transpose_read(const range<int>& this_box, int dim1, int dim2) {
	std::vector<cmplx> data(this_box.volume());
	assert(cmplx_mybox[dim2].contains(this_box));
	auto tbox = this_box.transpose(dim1, dim2);
	std::array<int, NDIM> i;
	for (i[0] = this_box.begin[0]; i[0] != this_box.end[0]; i[0]++) {
		for (i[1] = this_box.begin[1]; i[1] != this_box.end[1]; i[1]++) {
			for (i[2] = this_box.begin[2]; i[2] != this_box.end[2]; i[2]++) {
				auto j = i;
				std::swap(j[dim1], j[dim2]);
				data[tbox.index(j)] = Y[cmplx_mybox[dim2].index(i)];
			}
		}
	}
	return std::move(data);
}

static void split_box(range<int> box, std::vector<range<int>>& real_boxes) {
	if (box.volume() < MAX_BOX_SIZE) {
		real_boxes.push_back(box);
	} else {
		auto children = box.split();
		split_box(children.first, real_boxes);
		split_box(children.second, real_boxes);
	}
}
