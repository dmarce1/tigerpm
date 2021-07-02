/*
 * kick_pme.hpp
 *
 *  Created on: Jun 30, 2021
 *      Author: dmarce1
 */

#ifndef KICK_PME_HPP_
#define KICK_PME_HPP_


#include <tigerpm/tigerpm.hpp>
#include <tigerpm/range.hpp>

void kick_pme(range<int> box, int min_rung, double scale, double t0, bool first_call);
void kick_pme_begin(int min_rung, double scale, double t0, bool first_call);
void kick_pme_end();

#endif /* KICK_PME_HPP_ */
