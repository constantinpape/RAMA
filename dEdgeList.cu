#include "dEdgeList.h"
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <ECLgraph.h>
#include "time_measure_util.h"
#include "utils.h"
#include "dCOO.h"

__global__ void map_nodes(const int num_edges, 
                            const int* const __restrict__ node_mapping, 
                            int* __restrict__ rows, 
                            int* __restrict__ cols,
                            const bool* affected_edges,
                            const bool* cc_size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;
    for (int e = tid; e < num_edges; e += num_threads)
    {
        int new_r = rows[e];
        int new_c = cols[e];
        rows[e] = node_mapping[new_r];
        cols[e] = node_mapping[new_c];
        affected_edges[e] = min(cc_size[new_r], cc_size[new_c]) > 1;
    }
}

struct is_same_edge
{
    __host__ __device__
        bool operator()(const thrust::tuple<int,int> e1, const thrust::tuple<int,int> e2)
        {
            if((thrust::get<0>(e1) == thrust::get<0>(e2)) && (thrust::get<1>(e1) == thrust::get<1>(e2)))
                return true;
            else
                return false;
        }
};

struct is_affected_edge
{
    __host__ __device__
        bool operator()(const thrust::tuple<int,int,float,bool> e)
        {
            return thrust::get<3>(e);
        }
};

dEdgeList dEdgeList::contract_cuda(cusparseHandle_t handle, const thrust::device_vector<int>& node_mapping)
{
    MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME;

    // Get size of each connected component:
    thrust::constant_iterator<int> ones(1);
    assert(thrust::is_sorted(node_mapping.begin(), node_mapping.end()));
    thrust::device_vector<int> cc_ids(node_mapping.size());
    thrust::device_vector<int> cc_size(node_mapping.size());
    auto cc_last = thrust::reduce_by_key(node_mapping.begin(), node_mapping.end(), ones.begin(), cc_ids.begin(), cc_size.begin());
    int nr_output_nodes = std::distance(cc_ids.begin(), cc_last.first);
    cc_ids.resize(nr_output_nodes);
    cc_size.resize(nr_output_nodes);
    assert(thrust::is_sorted(cc_ids.begin(), cc_ids.end()));
    assert(cc_ids[0] = 0); //  cc_ids should now be a sequence starting from 0.
    assert(cc_ids[nr_output_nodes - 1] = nr_output_nodes - 1);

    const int numThreads = 256;

    thrust::device_vector<int> new_row_ids = row_ids;
    thrust::device_vector<int> new_col_ids = col_ids;
    thrust::device_vector<float> new_data = data;
    thrust::device_vector<bool> affected_edges(data.size());

    int num_edges = new_row_ids.size();
    int numBlocks = ceil(num_edges / (float) numThreads);

    map_nodes<<<numBlocks, numThreads>>>(num_edges, 
                                    thrust::raw_pointer_cast(node_mapping.data()), 
                                    thrust::raw_pointer_cast(new_row_ids.data()), 
                                    thrust::raw_pointer_cast(new_col_ids.data()),
                                    thrust::raw_pointer_cast(affected_edges.data()),
                                    thrust::raw_pointer_cast(cc_size.data()));

    auto first = thrust::make_zip_iterator(thrust::make_tuple(new_row_ids.begin(), new_col_ids.begin(), new_data.begin(), affected_edges.begin()));
    auto last = thrust::make_zip_iterator(thrust::make_tuple(new_row_ids.end(), new_col_ids.end(), new_data.end(), affected_edges.end()));

    // partition to unaffected edges for which nothing else needs to be done and affected edges
    // which might contain duplicates and would require to be reduced to a single edge by summation.
    auto first_unaff_edge = thrust::partition(first, last, is_affected_edge());
    const size_t nr_aff_edges = std::distance(first, first_unaff_edge);
    const size_t nr_unaffected_edges = std::distance(first_unaff_edge, last);

    thrust::device_vector<int m_rows(new_row_ids.begin(), new_row_ids.begin() + nr_aff_edges);
    thrust::device_vector<int m_cols(new_col_ids.begin(), new_col_ids.begin() + nr_aff_edges);
    thrust::device_vector<int m_data(new_data_ids.begin(), new_data_ids.begin() + nr_aff_edges);

    // sort the affected edges for reduce_by_key:
    coo_sorting(handle, m_cols, m_rows, m_data);
    first = thrust::make_zip_iterator(thrust::make_tuple(m_rows.begin(), m_cols.begin()));
    last = thrust::make_zip_iterator(thrust::make_tuple(m_rows.end(), m_cols.end()));

    thrust::device_vector<int> m_rows_red(nr_aff_edges);
    thrust::device_vector<int> m_cols_red(nr_aff_edges);
    auto first_output = thrust::make_zip_iterator(thrust::make_tuple(m_rows_red.begin(), m_cols_red.begin()));
    thrust::device_vector<float> m_data_red(nr_aff_edges);
    
    auto new_end = thrust::reduce_by_key(first, last, m_data.begin(), first_output, m_data_red.begin(), is_same_edge());
    int nr_aff_red_edges = std::distance(m_data.begin(), new_end.second);

    // allocate output:
    thrust::device_vector<int> out_rows(nr_aff_red_edges + nr_unaffected_edges);
    thrust::device_vector<int> out_cols(nr_aff_red_edges + nr_unaffected_edges);
    thrust::device_vector<int> out_data(nr_aff_red_edges + nr_unaffected_edges);

    // merge:
    thrust::copy(m_rows_red.begin(), m_rows_red.begin() + nr_aff_red_edges, out_rows.begin());
    thrust::copy(m_cols_red.begin(), m_cols_red.begin() + nr_aff_red_edges, out_cols.begin());
    thrust::copy(m_data_red.begin(), m_data_red.begin() + nr_aff_red_edges, out_data.begin());

    thrust::copy(new_row_ids.begin() + nr_aff_edges, new_row_ids.last(), out_rows.begin() + nr_aff_red_edges);
    thrust::copy(new_col_ids.begin() + nr_aff_edges, new_col_ids.last(), out_cols.begin() + nr_aff_red_edges);
    thrust::copy(new_data.begin() + nr_aff_edges, new_data.last(), out_data.begin() + nr_aff_red_edges);

    return dEdgeList(handle,
                out_cols.begin(), out_cols.end(),
                out_rows.begin(), out_rows.end(), 
                out_data.begin(), out_data.end());
}

struct is_diagonal
{
    __host__ __device__
        bool operator()(thrust::tuple<int,int,float> t)
        {
            return thrust::get<0>(t) == thrust::get<1>(t);
        }
};

void dEdgeList::remove_diagonal(cusparseHandle_t handle)
{
     auto begin = thrust::make_zip_iterator(thrust::make_tuple(col_ids.begin(), row_ids.begin(), data.begin()));
     auto end = thrust::make_zip_iterator(thrust::make_tuple(col_ids.end(), row_ids.end(), data.end()));

     auto new_last = thrust::remove_if(begin, end, is_diagonal());
     int new_num_edges = std::distance(begin, new_last);
     col_ids.resize(new_num_edges);
     row_ids.resize(new_num_edges);
     data.resize(new_num_edges);
}

struct diag_func
{
    float* d;
    __host__ __device__
        void operator()(thrust::tuple<int,int,float> t)
        {
            if(thrust::get<0>(t) == thrust::get<1>(t))
            {
                assert(d[thrust::get<0>(t)] == 0.0);
                d[thrust::get<0>(t)] = thrust::get<2>(t);
            }
        }
};

thrust::device_vector<float> dEdgeList::diagonal(cusparseHandle_t handle) const
{
    assert(rows() == cols());
    thrust::device_vector<float> d(rows(), 0.0);

    auto begin = thrust::make_zip_iterator(thrust::make_tuple(col_ids.begin(), row_ids.begin(), data.begin()));
    auto end = thrust::make_zip_iterator(thrust::make_tuple(col_ids.end(), row_ids.end(), data.end()));

    thrust::for_each(begin, end, diag_func({thrust::raw_pointer_cast(d.data())})); 

    return d;
}

float dEdgeList::sum()
{
    return thrust::reduce(data.begin(), data.end(), (float) 0.0, thrust::plus<float>());
}