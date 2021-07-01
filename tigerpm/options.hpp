
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
	double GM;
	double hsoft;
	double box_size;
	double rs;

	std::string config_file;
	std::string test;

	template<class A>
	void serialize(A&& arc, unsigned) {
		arc & hsoft;
		arc & eta;
		arc & GM;
		arc & parts_o_four;
		arc & parts_o_chain;
		arc & four_o_chain;
		arc & chain_dim;
		arc & four_dim;
		arc & parts_dim;
		arc & box_size;
		arc & config_file;
		arc & test;
	}
};

bool process_options(int argc, char *argv[]);
const options& get_options();
