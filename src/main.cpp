#include <tigerpm/tigerpm.hpp>
#include <tigerpm/hpx.hpp>
#include <tigerpm/options.hpp>
#include <tigerpm/test.hpp>
#include <tigerpm/fixed.hpp>
#include <tigerpm/fixedcapvec.hpp>
#include <tigerpm/stack_vector.hpp>

int hpx_main(int argc, char *argv[]) {
	hpx_init();

	process_options(argc,argv);
	if( get_options().test != "" ) {
		run_test(get_options().test);
	}

	return hpx::finalize();
}

#ifdef USE_HPX
int main(int argc, char *argv[]) {
	std::vector<std::string> cfg = {"hpx.commandline.allow_unknown=1"};
	cfg.push_back("hpx.stacks.small_size=262144");
	hpx::init(argc, argv, cfg);
}

#endif
