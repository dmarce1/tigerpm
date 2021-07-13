
#pragma once

#include <string>

struct options {

	int four_dim;
	int chain_dim;
	int parts_dim;
	int parts_o_four;
	int parts_o_chain;
	int four_o_chain;

	double eta;
	double sigma8;
	double code_to_cm;
	double code_to_s;
	double code_to_g;
	double hubble;
	double GM;
	double hsoft;
	double box_size;
	double rs;
	double omega_m;
	double omega_r;
	double z0;

	std::string config_file;
	std::string test;

	template<class A>
	void serialize(A&& arc, unsigned) {
		arc & sigma8;
		arc & hsoft;
		arc & eta;
		arc & GM;
		arc & code_to_cm;
		arc & hubble;
		arc & parts_o_four;
		arc & parts_o_chain;
		arc & four_o_chain;
		arc & chain_dim;
		arc & four_dim;
		arc & parts_dim;
		arc & box_size;
		arc & config_file;
		arc & test;
		arc & omega_m;
		arc & z0;
	}
};

bool process_options(int argc, char *argv[]);
const options& get_options();
