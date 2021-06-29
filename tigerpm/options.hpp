
#pragma once

#include <string>

struct options {

	int chain_dim;
	int parts_dim;
	int four_dim;

	double box_size;

	std::string config_file;
	std::string test;

	template<class A>
	void serialize(A&& arc, unsigned) {
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
