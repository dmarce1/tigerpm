
#pragma once

#include <string>

struct options {

	int chain_dim;
	int part_dim;

	std::string config_file;
	std::string test;

	template<class A>
	void serialize(A&& arc, unsigned) {
		arc & chain_dim;
		arc & part_dim;
		arc & config_file;
		arc & test;
	}
};

bool process_options(int argc, char *argv[]);
const options& get_options();
