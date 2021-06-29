#pragma once

#include <tigerpm/tigerpm.hpp>

#include <memory>

struct chaincell {
	int pbegin;
	int pend;
};

void chainmesh_create();
void chainmesh_exchange_boundaries();
