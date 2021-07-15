/*
 * checkpoint.hpp
 *
 *  Created on: Jul 15, 2021
 *      Author: dmarce1
 */

#ifndef CHECKPOINT_HPP_
#define CHECKPOINT_HPP_

#include <tigerpm/tigerpm.hpp>
#include <tigerpm/time.hpp>

struct driver_params {
	double a;
	double tau;
	double tau_max;
	double cosmicK;
	double esum0;
	int iter;
	size_t total_processed;
	double runtime;
	time_type itime;
};

void write_checkpoint(driver_params params);
void read_checkpoint(driver_params& params, int checknum);


#endif /* CHECKPOINT_HPP_ */
