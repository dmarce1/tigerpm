/*
 * fmmpm.hpp
 *
 *  Created on: Jul 9, 2021
 *      Author: dmarce1
 */

#ifndef FMMPM_HPP_
#define FMMPM_HPP_

#include <tigerpm/tree.hpp>

void kick_fmmpm(vector<tree> trees, range<int> box, int min_rung, double scale, double t0, bool first_call);
void kick_fmmpm_begin(int min_rung, double scale, double t0, bool first_call);
void kick_fmmpm_end();

#endif /* FMMPM_HPP_ */
