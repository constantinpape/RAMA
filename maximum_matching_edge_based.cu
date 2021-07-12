#include <iostream>
#include <thrust/device_vector.h>
#include <cuda_runtime.h>
#include "maximum_matching_edge_based.h"
#include "time_measure_util.h"
#include "utils.h"

#define numThreads 256

__device__ static float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__global__ void pick_best_edge(const int num_edges, 
                            const int* const __restrict__ row_ids, 
                            const int* const __restrict__ col_ids, 
                            const int* const __restrict__ costs_quant, 
                            const bool* const __restrict__ v_matched,
                            int* __restrict__ highest_edge_index, 
                            int* __restrict__ highest_edge_value)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;

    for (int edge = tid; edge < num_edges; edge += num_threads) 
    {
        int r = row_ids[edge];
        int c = col_ids[edge];
        if (!v_matched[r] && !v_matched[c])
        {
            int w = costs_quant[edge];
            atomicMax(&highest_edge_value[r], w);
            if (highest_edge_value[r] == w)
                highest_edge_index[r] = edge;
            // other direction:
            atomicMax(&highest_edge_value[c], w);
            if (highest_edge_value[c] == w)
                highest_edge_index[c] = edge;
        }
        __syncthreads(); 
    }
}

__global__ void mark_matches(const int num_edges, 
                            const int* const __restrict__ row_ids, 
                            const int* const __restrict__ col_ids, 
                            const int* const __restrict__ highest_edge_index,
                            bool* __restrict__ matched_edges,
                            bool* __restrict__ v_matched)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;

    for (int edge = tid; edge < num_edges; edge += num_threads) 
    {
        int r = row_ids[edge];
        int c = col_ids[edge];
        if (highest_edge_index[c] >= 0 && highest_edge_index[c] == highest_edge_index[r])
        {
            matched_edges[edge] = true;
            v_matched[r] = true;
            v_matched[c] = true;
        }
        __syncthreads(); 
    }
}

struct is_unmatched {
    __host__ __device__
        inline int operator()(const thrust::tuple<int,int,bool> e)
        {
            return !thrust::get<2>(e);
        }
};

struct cost_scaling_func {
    const float scaling_factor;
    __host__ __device__
        inline int operator()(const float x)
        {
            return int(scaling_factor * x);
        } 
};

std::tuple<thrust::device_vector<int>, thrust::device_vector<int>> filter_edges_by_matching_edge_based(cusparseHandle_t handle, const dCOO& A)
{
    thrust::device_vector<int> row_ids_directed, col_ids_directed;
    thrust::device_vector<float> costs_directed;
    std::tie(row_ids_directed, col_ids_directed, costs_directed) = to_directed(A.get_row_ids(), A.get_col_ids(), A.get_data());

    const float scaling_factor = 1e6 / *thrust::max_element(costs_directed.begin(), costs_directed.end());
    thrust::device_vector<int> costs_directed_quant(costs_directed.size());
    thrust::transform(costs_directed.begin(), costs_directed.end(), costs_directed_quant.begin(), cost_scaling_func({scaling_factor}));

    const int num_dir_edges = row_ids_directed.size();
    //TODO: Shuffle edge indices so that concurrent threads have less chance of writing onto the same vertex due to atomicMax.
    thrust::device_vector<int> highest_edge_index(A.rows());
    thrust::device_vector<int> highest_edge_value(A.rows(), 0.0f);
    int numBlocks = ceil(num_dir_edges / (float) numThreads);

    thrust::device_vector<bool> matched_edges(num_dir_edges, false);
    thrust::device_vector<bool> v_matched(A.rows(), false);
    for (int t = 0; t < 2; t++)
    {
        thrust::fill(highest_edge_index.begin(), highest_edge_index.end(), -1);
        thrust::fill(highest_edge_value.begin(), highest_edge_value.end(), 0);

        pick_best_edge<<<numBlocks, numThreads>>>(num_dir_edges, 
                                                    thrust::raw_pointer_cast(row_ids_directed.data()), 
                                                    thrust::raw_pointer_cast(col_ids_directed.data()), 
                                                    thrust::raw_pointer_cast(costs_directed_quant.data()),
                                                    thrust::raw_pointer_cast(v_matched.data()),
                                                    thrust::raw_pointer_cast(highest_edge_index.data()),
                                                    thrust::raw_pointer_cast(highest_edge_value.data()));

        mark_matches<<<numBlocks, numThreads>>>(num_dir_edges, 
            thrust::raw_pointer_cast(row_ids_directed.data()), 
            thrust::raw_pointer_cast(col_ids_directed.data()), 
            thrust::raw_pointer_cast(highest_edge_index.data()),
            thrust::raw_pointer_cast(matched_edges.data()),
            thrust::raw_pointer_cast(v_matched.data()));

        std::cout << "matched sum = " << thrust::reduce(v_matched.begin(), v_matched.end(), 0) << "\n";
    }

    auto first_m = thrust::make_zip_iterator(thrust::make_tuple(row_ids_directed.begin(), col_ids_directed.begin(), matched_edges.begin()));
    auto last_m = thrust::make_zip_iterator(thrust::make_tuple(row_ids_directed.end(), col_ids_directed.end(), matched_edges.end()));
    auto matched_last = thrust::remove_if(first_m, last_m, is_unmatched());
    const int nr_matched_edges = std::distance(first_m, matched_last);
    row_ids_directed.resize(nr_matched_edges);
    col_ids_directed.resize(nr_matched_edges);

    std::cout << "# vertices = " << A.rows() << "\n";
    std::cout << "# matched edges = " << nr_matched_edges << " / "<< A.edges() / 2 << "\n";

    return to_undirected(row_ids_directed, col_ids_directed);
}