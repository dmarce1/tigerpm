#pragma once

#include <tigerpm/tigerpm.hpp>
#include <tigerpm/range.hpp>

#include <memory>

struct chaincell {
	int pbegin;
	int pend;
};

void chainmesh_create();
void chainmesh_exchange_begin();
void chainmesh_exchange_end();
range<int> chainmesh_interior_box();
chaincell chainmesh_get(const array<int,NDIM>&);
