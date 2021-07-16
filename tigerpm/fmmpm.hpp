/*
 * fmmpm.hpp
 *
 *  Created on: Jul 9, 2021
 *      Author: dmarce1
 */

#ifndef FMMPM_HPP_
#define FMMPM_HPP_

#include <tigerpm/tree.hpp>

struct kick_return {
	int max_rung;
	double flops;
	double pot;
	double fx;
	double fy;
	double fz;
	double fnorm;
	double pp;
	double pc;
	double cp;
	double cc;
	size_t nactive;
	template<class Arc>
	void serialize(Arc& a, unsigned) {
		a & max_rung;
		a & flops;
		a & pot;
		a & fx;
		a & fy;
		a & fz;
		a & fnorm;
		a & nactive;
		a & pp;
		a & pc;
		a & cp;
		a & cc;
	}
};

kick_return kick_fmmpm(vector<tree> trees, range<int> box, int min_rung, double scale, double t0, double theta, bool first_call, bool full_eval, kick_return* =
		nullptr);
kick_return kick_fmmpm_begin(int min_rung, double scale, double t0, double theta, bool first_call, bool full_eval);
void kick_fmmpm_end();

#endif /* FMMPM_HPP_ */
