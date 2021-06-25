#include <tigerpm/test.hpp>
#include <tigerpm/fft.hpp>
#include <tigerpm/timer.hpp>

static void fft_test();

void run_test(std::string test) {
	printf("Testing\n");
	if (test == "fft") {
		fft_test();
	} else {
		PRINT("%s is an unknown test.\n", test.c_str());
	}
	printf("Test complete\n");
}

static void fft_test() {
	PRINT("Doing FFT test\n");
	const int N = 4;
	fft3d_init(N);
	std::vector<float> R(N * N * N);
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			for (int k = 0; k < N; k++) {
				R[i * N * N + j * N + k] = sin(2.0 * M_PI * double(k) / N);
			}
		}
	}
	range<int> box(N);
	fft3d_accumulate(box, std::move(R));
	timer tm;
	tm.start();
	fft3d_execute();
	tm.stop();
	box.end[2] = N / 2 + 1;
	const auto Y = fft3d_read_complex(box);
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			for (int k = 0; k < N / 2 + 1; k++) {
				const int l = (i * N + j) * (N / 2 + 1) + k;
				PRINT("%i %i %i %e %e\n", i, j, k, Y[l].real(), Y[l].imag());
			}
		}
	}

	PRINT("FFT took %e seconds\n", tm.read());
	fft3d_destroy();
}
