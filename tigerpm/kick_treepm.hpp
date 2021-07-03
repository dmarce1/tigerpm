/*
 * kick_treepm.hpp
 *
 *  Created on: Jul 2, 2021
 *      Author: dmarce1
 */

#ifndef KICK_TREEPM_HPP_
#define KICK_TREEPM_HPP_

#include <tigerpm/tree.hpp>

void kick_treepm_begin(int min_rung, double scale, double t0, bool first_call);
void kick_treepm(vector<tree>& trees, vector<vector<sink_bucket>>& buckets, range<int> box, int min_rung, double scale, double t0, bool first_call);
void kick_treepm_end();



#endif /* KICK_TREEPM_HPP_ */
