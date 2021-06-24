#include <tigerpm/test.hpp>
#include <tigerpm/fft.hpp>

static void fft_test();

void run_test(std::string test) {
	if (test == "fft") {
		fft_test();
	} else {
		PRINT("%s is an unknown test.\n", test.c_str());
	}
}

static void fft_test() {
	PRINT("Doing FFT test\n");
	const int N = 10;
	fft3d_init(N);
	std::vector<float> Y(N * N * N);
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			for (int k = 0; k < N; k++) {
				Y[i * N * N + j * N + k] = i;
			}
		}
	}
	range<int> box(N);
	fft3d_accumulate(box,std::move(Y));
	fft3d_execute();
	Y = fft3d_read(box);
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			for (int k = 0; k < N; k++) {
				PRINT( "%i %i %i %e\n", i, j, k, Y[i * N * N + j * N + k]);
			}
		}
	}
	fft3d_destroy();
}
