/*
 * options.cpp
 *
 *  Created on: Jan 25, 2021
 *      Author: dmarce1
 */

#include <tigerpm/tigerpm.hpp>
#include <tigerpm/hpx.hpp>
#include <tigerpm/options.hpp>

options global_opts;

static void set_options(const options& opts);

HPX_PLAIN_ACTION(set_options);


const options& get_options() {
	return global_opts;
}

static void set_options(const options& opts) {
	std::vector<hpx::future<void>> futs;
	for (auto c : hpx_children()) {
		futs.push_back(hpx::async < set_options_action > (c, opts));
	}
	global_opts = opts;
	hpx::wait_all(futs.begin(), futs.end());
}

bool process_options(int argc, char *argv[]) {
	options opts;
	namespace po = hpx::program_options;
	bool rc;
	po::options_description command_opts("options");

	command_opts.add_options()                                                                       //
	("help", "produce help message")                                                                 //
	("config_file", po::value<std::string>(&(opts.config_file))->default_value(""), "configuration file") //
	("part_dim", po::value<int>(&(opts.part_dim))->default_value(1000), "nparts^(1/2)") //
	("chain_dim", po::value<int>(&(opts.chain_dim))->default_value(250), "chain mesh dimension size") //
	("test", po::value<std::string>(&(opts.test))->default_value(""), "test problem") //
			;

	hpx::program_options::variables_map vm;
	po::store(po::parse_command_line(argc, argv, command_opts), vm);
	po::notify(vm);
	if (vm.count("help")) {
		std::cout << command_opts << "\n";
		rc = false;
	} else {
		if (!opts.config_file.empty()) {
			std::ifstream cfg_fs { vm["config_file"].as<std::string>() };
			if (cfg_fs) {
				po::store(po::parse_config_file(cfg_fs, command_opts), vm);
				rc = true;
			} else {
				PRINT("Configuration file %s not found!\n", opts.config_file.c_str());
				return false;
			}
		} else {
			rc = true;
		}
	}

	if (rc) {
		po::notify(vm);
	}


#define SHOW( opt ) PRINT( "%s = %e\n",  #opt, (double) opts.opt)
#define SHOW_STRING( opt ) std::cout << std::string( #opt ) << " = " << opts.opt << '\n';

	PRINT("Simulation Options\n");

	SHOW(chain_dim);
	SHOW(part_dim);
	SHOW_STRING(config_file);
	SHOW_STRING(test);

	set_options(opts);

	return rc;
}

