#include <vector>
#include <tuple>
#include <set>
#include <random>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "rama_utils.h"
#include <thrust/count.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/binary_search.h>

namespace py=pybind11;

std::tuple<std::vector<int>, std::vector<int>> compute_incorrect_edge_indices(
    const std::vector<int>& node_labels_a,
    const std::vector<int>& node_labels_b) 
{
    const int num_nodes = node_labels_a.size();
    std::vector<int> edge_i, edge_j;
    edge_i.reserve(num_nodes);
    edge_j.reserve(num_nodes);
    for (int i = 0; i != num_nodes; ++i)
    {
        const int label_a_i = node_labels_a[i];
        const int label_b_i = node_labels_b[i];
        for (int j = i; j != num_nodes; ++j)
        {
            const int label_a_j = node_labels_a[j];
            const int label_b_j = node_labels_b[j];
            const bool edge_cut_a = label_a_i != label_a_j;
            const bool edge_cut_b = label_b_i != label_b_j;
            if ((edge_cut_a && !edge_cut_b) || (!edge_cut_a && edge_cut_b))
            {
                edge_i.push_back(i);
                edge_j.push_back(j);
            }
        }
    }
	return {edge_i, edge_j};
}

std::tuple<std::vector<int>, std::vector<int>> compute_incorrect_edge_indices_sampled(
    const std::vector<int>& node_labels_a, const std::vector<int>& node_labels_b, const float subsampling_factor) 
{
    if (subsampling_factor < 1)
        throw std::runtime_error("subsampling_factor should be >= 1");
    if (subsampling_factor == 1)
        return compute_incorrect_edge_indices(node_labels_a, node_labels_b);

    const int num_nodes = node_labels_a.size();
    const int num_sampled_nodes = num_nodes / subsampling_factor;
    std::vector<int> edge_i, edge_j;
    edge_i.reserve(num_nodes);
    edge_j.reserve(num_nodes);
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> uni(0, num_nodes - 1);
    for (int s1 = 0; s1 != num_sampled_nodes; ++s1)
    {
        auto i = uni(rng);
        const int label_a_i = node_labels_a[i];
        const int label_b_i = node_labels_b[i];
        for (int s2 = 0; s2 != 2 * num_sampled_nodes; ++s2)
        {
            auto j = uni(rng);
            if (j <= i)
                continue;
            const int label_a_j = node_labels_a[j];
            const int label_b_j = node_labels_b[j];
            const bool edge_cut_a = label_a_i != label_a_j;
            const bool edge_cut_b = label_b_i != label_b_j;
            if ((edge_cut_a && !edge_cut_b) || (!edge_cut_a && edge_cut_b))
            {
                edge_i.push_back(i);
                edge_j.push_back(j);
            }
        }
    }
	return {edge_i, edge_j};
}

struct lower_triangle_size_wo_diagonal
{
    __device__
    void operator()(long long int& n) const
    {
        n = (n * (n - 1)) / 2;
    }
};

// from: https://stackoverflow.com/questions/40950460/how-to-convert-triangular-matrix-indexes-in-to-row-column-coordinates
__device__ void lower_triangle_indices(const unsigned long int index, unsigned long int& x, unsigned long int& y)
{
    y = ceil(sqrt((2.0 * index + 1) + 0.25) - 0.5);
    x = index - (y - 1) * y / 2;
}

struct edge_labels_mismatch_func
{
    const long long int* cum_num_threads_per_label;
    const int* cum_num_nodes_per_label_p;
    const int* label_to_node_map_p;
    const int* node_labels_q;
    const int num_labels_p;

    __device__
    bool operator()(const long long int edge_index) const
    {
        const long long int* label_loc = thrust::upper_bound(thrust::seq, cum_num_threads_per_label, cum_num_threads_per_label + num_labels_p, edge_index);
        const int label_index = thrust::distance(cum_num_threads_per_label, label_loc);
        assert(label_index >= 0 && label_index < num_labels_p);
        const int prev_num_threads = (label_index > 0) ? cum_num_threads_per_label[label_index - 1] : 0;
        const int local_edge_index = edge_index - prev_num_threads;

        unsigned long int local_x, local_y;
        lower_triangle_indices(local_edge_index, local_x, local_y); // Convert local_edge_index to 2d index to a lower triangular matrix i.e. (local_x < local_y).
        const int start_offset = (label_index > 0) ? cum_num_nodes_per_label_p[label_index - 1] : 0;
        const int node_i = label_to_node_map_p[start_offset + local_x];
        const int node_j = label_to_node_map_p[start_offset + local_y];
        const int label_i_in_q = node_labels_q[node_i];
        const int label_j_in_q = node_labels_q[node_j];
        return label_i_in_q != label_j_in_q;
    }
};

struct put_mismatched_edges_func
{
    const long long int* cum_num_threads_per_label;
    const int* cum_num_nodes_per_label_p;
    const int* label_to_node_map_p;
    const int* node_labels_q;
    const int num_labels_p;
    int* edge_i;
    int* edge_j;

    __device__
    void operator()(const thrust::tuple<long long int, long long int> t) const
    {
        const long long int edge_index = thrust::get<0>(t);
        const long long int out_index = thrust::get<1>(t);
        const long long int* label_loc = thrust::upper_bound(thrust::seq, cum_num_threads_per_label, cum_num_threads_per_label + num_labels_p, edge_index);
        const int label_index = thrust::distance(cum_num_threads_per_label, label_loc);
        const int prev_num_threads = (label_index > 0) ? cum_num_threads_per_label[label_index - 1] : 0;
        const int local_edge_index = edge_index - prev_num_threads;

        unsigned long int local_x, local_y;
        lower_triangle_indices(local_edge_index, local_x, local_y); // Convert local_edge_index to 2d index to a lower triangular matrix i.e. (local_x < local_y).
        const int start_offset = (label_index > 0) ? cum_num_nodes_per_label_p[label_index - 1] : 0;
        const int node_i = label_to_node_map_p[start_offset + local_x];
        const int node_j = label_to_node_map_p[start_offset + local_y];

        edge_i[out_index] = min(node_i, node_j);
        edge_j[out_index] = max(node_i, node_j);
    }
};

std::tuple<thrust::device_vector<int>, thrust::device_vector<int>> find_joined_in_p_cut_in_q_thrust(const thrust::device_vector<int>& node_labels_p, const thrust::device_vector<int>& node_labels_q)
{
    thrust::device_vector<int> node_indices_p(node_labels_p.size());
    thrust::sequence(node_indices_p.begin(), node_indices_p.end());
    thrust::device_vector<int> node_labels_p_sorted(node_labels_p);
    
    auto first = thrust::make_zip_iterator(thrust::make_tuple(node_labels_p_sorted.begin(), node_indices_p.begin()));
    auto last = thrust::make_zip_iterator(thrust::make_tuple(node_labels_p_sorted.end(), node_indices_p.end()));
    thrust::sort(first, last);

    thrust::device_vector<int> cum_num_nodes_per_label_p(node_labels_p_sorted.size());
    auto last_num = thrust::reduce_by_key(node_labels_p_sorted.begin(), node_labels_p_sorted.end(), thrust::make_constant_iterator<long long int>(1), 
                                        thrust::make_discard_iterator(), cum_num_nodes_per_label_p.begin());
    const int num_labels_p = thrust::distance(cum_num_nodes_per_label_p.begin(), last_num.second);
    cum_num_nodes_per_label_p.resize(num_labels_p);

    thrust::device_vector<long long int> cum_num_edges_per_label_p(cum_num_nodes_per_label_p);
    thrust::inclusive_scan(cum_num_nodes_per_label_p.begin(), cum_num_nodes_per_label_p.end(), cum_num_nodes_per_label_p.begin());

    thrust::for_each(cum_num_edges_per_label_p.begin(), cum_num_edges_per_label_p.end(), lower_triangle_size_wo_diagonal());

    thrust::inclusive_scan(cum_num_edges_per_label_p.begin(), cum_num_edges_per_label_p.end(), cum_num_edges_per_label_p.begin());
    const long long int total_num_edges = cum_num_edges_per_label_p[cum_num_edges_per_label_p.size() - 1];

    edge_labels_mismatch_func compute_num_mismatch({thrust::raw_pointer_cast(cum_num_edges_per_label_p.data()),
                                                    thrust::raw_pointer_cast(cum_num_nodes_per_label_p.data()),
                                                    thrust::raw_pointer_cast(node_indices_p.data()),
                                                    thrust::raw_pointer_cast(node_labels_q.data())});

    const long long int num_mismatch = thrust::count_if(thrust::make_counting_iterator<long long int>(0), thrust::make_counting_iterator<long long int>(0) + total_num_edges, compute_num_mismatch);
    thrust::device_vector<int> mismatched_edge_indices(num_mismatch);

    thrust::copy_if(thrust::make_counting_iterator<long long int>(0), thrust::make_counting_iterator<long long int>(0) + total_num_edges, mismatched_edge_indices.begin(), compute_num_mismatch);

    thrust::device_vector<int> edge_i(num_mismatch);
    thrust::device_vector<int> edge_j(num_mismatch);
    put_mismatched_edges_func put_mismatch({thrust::raw_pointer_cast(cum_num_edges_per_label_p.data()),
                                            thrust::raw_pointer_cast(cum_num_nodes_per_label_p.data()),
                                            thrust::raw_pointer_cast(node_indices_p.data()),
                                            thrust::raw_pointer_cast(node_labels_q.data()),
                                            num_labels_p,
                                            thrust::raw_pointer_cast(edge_i.data()),
                                            thrust::raw_pointer_cast(edge_j.data())});

    auto first_edge = thrust::make_zip_iterator(thrust::make_tuple(mismatched_edge_indices.begin(), thrust::make_counting_iterator<long long int>(0)));
    auto last_edge = thrust::make_zip_iterator(thrust::make_tuple(mismatched_edge_indices.end(), thrust::make_counting_iterator<long long int>(0) + num_mismatch));

    thrust::for_each(first_edge, last_edge, put_mismatch);
    return {edge_i, edge_j};
}

std::tuple<std::vector<int>, std::vector<int>> compute_incorrect_edge_indices_thrust(
    const std::vector<int>& node_labels_a,
    const std::vector<int>& node_labels_b) 
{
    const thrust::device_vector<int> dev_node_labels_a(node_labels_a);
    const thrust::device_vector<int> dev_node_labels_b(node_labels_b);

    thrust::device_vector<int> edge_i, edge_j;
    std::tie(edge_i, edge_j) = find_joined_in_p_cut_in_q_thrust(node_labels_a, node_labels_b);

    std::vector<int> h_edge_i(edge_i.size());
    std::vector<int> h_edge_j(edge_j.size());

    thrust::copy(edge_i.begin(), edge_i.end(), h_edge_i.begin());
    thrust::copy(edge_j.begin(), edge_j.end(), h_edge_j.begin());

    std::tie(edge_i, edge_j) = find_joined_in_p_cut_in_q_thrust(node_labels_b, node_labels_a);
    std::vector<int> h_edge_i_full(edge_i.size() + h_edge_i.size());
    std::vector<int> h_edge_j_full(edge_j.size() + h_edge_j.size());

    std::copy(h_edge_i.begin(), h_edge_i.end(), h_edge_i_full.begin());
    std::copy(h_edge_j.begin(), h_edge_j.end(), h_edge_j_full.begin());

    thrust::copy(edge_i.begin(), edge_i.end(), h_edge_i_full.begin() + h_edge_i.size());
    thrust::copy(edge_j.begin(), edge_j.end(), h_edge_j_full.begin() + h_edge_j.size());
    return {h_edge_i_full, h_edge_j_full};
}

PYBIND11_MODULE(rand_index_py, m) {
    m.doc() = "Utils for computing Rand Index efficiently.";
    m.def("compute_incorrect_edge_indices", [](const std::vector<int>& node_labels_a, const std::vector<int>& node_labels_b) {
            return compute_incorrect_edge_indices(node_labels_a, node_labels_b);
            });
    m.def("compute_incorrect_edge_indices_sampled", [](const std::vector<int>& node_labels_a, const std::vector<int>& node_labels_b, const float subsampling_factor) {
        return compute_incorrect_edge_indices_sampled(node_labels_a, node_labels_b, subsampling_factor);
        });
    m.def("compute_incorrect_edge_indices_thrust", [](const std::vector<int>& node_labels_a, const std::vector<int>& node_labels_b) {
        return compute_incorrect_edge_indices_thrust(node_labels_a, node_labels_b);
        });
}

