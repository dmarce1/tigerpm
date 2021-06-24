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
	const int N = 100;
	fft3d_init(N);
	fft3d_execute();
	fft3d_destroy();
}
