#pragma once

#include <thrust/device_vector.h>
#include <cusparse.h>
#include "dCOO.h"

std::tuple<thrust::device_vector<int>, thrust::device_vector<int>, thrust::device_vector<int>> parallel_small_cycle_packing_cuda(cusparseHandle_t handle, const dCOO& A);