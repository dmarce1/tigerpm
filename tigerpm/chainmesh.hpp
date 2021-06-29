#pragma once

#include <tigerpm/tigerpm.hpp>

#include <memory>

struct chaincell {
	std::vector<int> particles;
	std::shared_ptr<std::atomic<int>> lock;
	chaincell()  {
		lock = std::make_shared<std::atomic<int>>(0);
	}
};

void chainmesh_create();
