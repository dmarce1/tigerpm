#include <hpx/hpx.hpp>

int hpx_main(int argc, char *argv[]) {

	return hpx::finalize();
}

int main(int argc, char *argv[]) {
	std::vector<std::string> cfg = {"hpx.commandline.allow_unknown=1"};
	cfg.push_back("hpx.stacks.small_size=2097152");
	hpx::init(argc, argv, cfg);
}


