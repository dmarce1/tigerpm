#include <tigerpm/test.hpp>
#include <tigerpm/fft.hpp>
#include <tigerpm/timer.hpp>
#include <tigerpm/particles.hpp>
#include <tigerpm/gravity_long.hpp>
#include <tigerpm/gravity_short.hpp>
#include <tigerpm/kick_pm.hpp>
#include <tigerpm/kick_pme.hpp>
#include <tigerpm/kick_treepm.hpp>
#include <tigerpm/chainmesh.hpp>
#include <tigerpm/initialize.hpp>

static void chainmesh_test();
static void kick_treepm_test();
static void kick_pm_test();
static void kick_pme_test();
static void fft_test();
static void particle_test();
static void gravity_long_test();
static void sort_test();
static void ic_test();

void run_test(std::string test) {
	printf("Testing\n");
	if (test == "fft") {
		fft_test();
	} else if (test == "ic") {
		ic_test();
	} else if (test == "chainmesh") {
		chainmesh_test();
	} else if (test == "sort") {
		sort_test();
	} else if (test == "parts") {
		particle_test();
	} else if (test == "kick_pme") {
		kick_pme_test();
	} else if (test == "treepm") {
		kick_treepm_test();
	} else if (test == "kick_pm") {
		kick_pm_test();
	} else if (test == "gravity_long") {
		gravity_long_test();
	} else {
		PRINT("%s is an unknown test.\n", test.c_str());
	}
	printf("Test complete\n");
}

static void ic_test() {
	initialize();
}

static void fft_test() {
	PRINT("Doing FFT test\n");
	const int N = 1290;
	fft3d_init(N);
	vector<float> R(N * N * N);
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			for (int k = 0; k < N; k++) {
				R[i * N * N + j * N + k] = 0.0;
			}
		}
	}
	R[0] = 1.0;
	range<int> box(N);
	fft3d_accumulate_real(box, std::move(R));
	timer tm;
	tm.start();
	fft3d_execute();
	tm.stop();
	PRINT("Fourier took %e seconds\n", tm.read());
	tm.reset();
	tm.start();
	fft3d_inv_execute();
	tm.stop();
	PRINT("Inverse Fourier took %e seconds\n", tm.read());
//	box.end[2] = N / 2 + 1;
	const auto Y = fft3d_read_real(box);
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			for (int k = 0; k < N / 2 + 1; k++) {
				const int l = (i * N + j) * (N / 2 + 1) + k;
//				PRINT("%i %i %i %e\n", i, j, k, Y[l]);
			}
		}
	}

	PRINT("FFT took %e seconds\n", tm.read());
	fft3d_destroy();
}

static void particle_test() {
	PRINT("Doing particle test\n");
	particles_random_init();
	timer tm;
	tm.start();
	particles_domain_sort();
	tm.stop();
	PRINT("Test took %e seconds\n", tm.read());
	tm.reset();
}

static void gravity_long_test() {
	timer tm1, tm2;
	particles_random_init();
	tm1.start();
	particles_domain_sort();
	tm1.stop();
	tm2.start();
	gravity_long_compute();
	tm2.stop();
	PRINT("%e s to sort, %e s to compute, %e total\n", tm1.read(), tm2.read(), tm1.read() + tm2.read());
}

static void kick_pm_test() {
	timer tm1, tm2, tm3, tm4;
	particles_random_init();
	PRINT("DOMAIN SORT\n");
	tm1.start();
	particles_domain_sort();
	tm1.stop();
	tm2.start();
	PRINT("FOURIER\n");
	gravity_long_compute();
	tm2.stop();
	PRINT("KICK\n");
	tm3.start();
	kick_pm();
	tm3.stop();
	PRINT("COMPARISON\n");
#ifdef FORCE_TEST
	tm4.start();
	gravity_short_ewald_compare(10000);
	tm4.stop();
#endif
	PRINT("%e s to sort, %e s to compute gravity, %e s to kick, %e s on comparison, %e total\n", tm1.read(), tm2.read(),
			tm3.read(), tm4.read(), tm1.read() + tm2.read() + tm3.read() + tm4.read());
}

static void chainmesh_test() {
	timer tm;
	double total = 0.0;
	particles_random_init();

	PRINT("DOMAIN SORT\n");
	tm.start();
	particles_domain_sort();
	tm.stop();
	PRINT("%e s\n", tm.read());
	total += tm.read();
	tm.reset();
	PRINT("\n");

	PRINT("SORT\n");
	tm.start();
	chainmesh_create();
	tm.stop();
	PRINT("%e s\n", tm.read());
	total += tm.read();
	tm.reset();
	PRINT("\n");

	PRINT("BOUNDARIES\n");
	tm.start();
	chainmesh_exchange_begin();
	chainmesh_exchange_end();
	tm.stop();
	PRINT("%e s\n", tm.read());
	total += tm.read();
	tm.reset();
	PRINT("\n");

	PRINT("%e s total\n", total);

}

static void kick_pme_test() {
	timer tm;
	particles_random_init();

	PRINT("DOMAIN SORT\n");
	tm.start();
	particles_domain_sort();
	tm.stop();
	PRINT("%e s\n", tm.read());
	tm.reset();

	PRINT("FOURIER\n");
	tm.start();
	gravity_long_compute(GRAVITY_LONG_PME);
	tm.stop();
	PRINT("%e s\n", tm.read());
	tm.reset();

	PRINT("SORT\n");
	tm.start();
	chainmesh_create();
	tm.stop();
	PRINT("%e s\n", tm.read());
	tm.reset();
	PRINT("\n");

	PRINT("BOUNDARIES\n");
	tm.start();
	chainmesh_exchange_begin();
	chainmesh_exchange_end();
	tm.stop();
	PRINT("%e s\n", tm.read());
	tm.reset();
	PRINT("\n");

	PRINT("KICK\n");
	tm.start();
	kick_pme_begin(0, 1.0, 1.0, true);
	kick_pme_end();
	tm.stop();
	PRINT("%e s\n", tm.read());
	tm.reset();

#ifdef FORCE_TEST
	PRINT("COMPARE\n");
	tm.start();
	gravity_short_ewald_compare(100);
	tm.stop();
	PRINT("%e s\n", tm.read());
	tm.reset();
#endif

}


static void kick_treepm_test() {
	timer tm;
	initialize();

	PRINT("DOMAIN SORT\n");
	tm.start();
	particles_domain_sort();
	tm.stop();
	PRINT("%e s\n", tm.read());
	tm.reset();

	PRINT("FOURIER\n");
	tm.start();
	gravity_long_compute(GRAVITY_LONG_PME);
	tm.stop();
	PRINT("%e s\n", tm.read());
	tm.reset();

	PRINT("SORT\n");
	tm.start();
	chainmesh_create();
	tm.stop();
	PRINT("%e s\n", tm.read());
	tm.reset();
	PRINT("\n");

	PRINT("BOUNDARIES\n");
	tm.start();
	chainmesh_exchange_begin();
	chainmesh_exchange_end();
	tm.stop();
	PRINT("%e s\n", tm.read());
	tm.reset();
	PRINT("\n");

	PRINT("KICK\n");
	tm.start();
	kick_treepm_begin(0, 1.0, 1.0, true);
	kick_treepm_end();
	tm.stop();
	PRINT("%e s\n", tm.read());
	tm.reset();

#ifdef FORCE_TEST
	PRINT("COMPARE\n");
	tm.start();
	gravity_short_ewald_compare(100);
	tm.stop();
	PRINT("%e s\n", tm.read());
	tm.reset();
#endif

}

static void sort_test() {
	particles_random_init();
	particles_domain_sort();
	timer tm;
	tm.start();
	chainmesh_create();
	tm.stop();
	PRINT("%e s\n", tm.read());
}
