#include <tigerpm/checkpoint.hpp>
#include <tigerpm/particles.hpp>

void write_checkpoint(driver_params params) {
	PRINT("Writing checkpoint\n");
	const std::string fname = std::string("checkpoint.") + std::to_string(params.iter) + std::string(".dat");
	FILE* fp = fopen(fname.c_str(), "wb");
	if (fp == nullptr) {
		PRINT("Unable to open %s for writing.\n", fname.c_str());
		abort();
	}
	fwrite(&params, sizeof(driver_params), 1, fp);
	particles_save(fp);

	fclose(fp);
}

void read_checkpoint(driver_params& params, int checknum) {
	PRINT("Reading checkpoint\n");
	const std::string fname = std::string("checkpoint.") + std::to_string(checknum) + std::string(".dat");
	FILE* fp = fopen(fname.c_str(), "rb");
	if (fp == nullptr) {
		PRINT("Unable to open %s for reading.\n", fname.c_str());
		abort();
	}
	FREAD(&params, sizeof(driver_params), 1, fp);
	particles_load(fp);

	fclose(fp);
}
