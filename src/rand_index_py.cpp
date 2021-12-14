#include <vector>
#include <tuple>
#include <set>
#include <random>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <omp.h>

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

std::tuple<std::vector<int>, std::vector<int>> compute_incorrect_edge_indices_helper(
    const std::vector<int>& node_labels_a, const std::vector<int>& node_labels_b, const int num_sampled_nodes_a, const int num_sampled_nodes_b)
{
    std::vector<int> edge_i, edge_j;
    const int num_nodes = node_labels_a.size();
    edge_i.reserve(num_nodes);
    edge_j.reserve(num_nodes);
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> uni(0, num_nodes - 1);
    for (int s1 = 0; s1 != num_sampled_nodes_a; ++s1)
    {
        auto i = uni(rng);
        const int label_a_i = node_labels_a[i];
        const int label_b_i = node_labels_b[i];
        for (int s2 = 0; s2 != 2 * num_sampled_nodes_b; ++s2)
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

std::tuple<std::vector<int>, std::vector<int>> compute_incorrect_edge_indices_sampled(
    const std::vector<int>& node_labels_a, const std::vector<int>& node_labels_b, const float subsampling_factor) 
{
    if (subsampling_factor < 1)
        throw std::runtime_error("subsampling_factor should be >= 1");
    if (subsampling_factor == 1)
        return compute_incorrect_edge_indices(node_labels_a, node_labels_b);

    const int num_nodes = node_labels_a.size();
    const int num_sampled_nodes = num_nodes / subsampling_factor;
    
    const int num_threads = 8; // omp_get_max_threads();
    std::vector<std::vector<int>> edge_i_comb(num_threads);
    std::vector<std::vector<int>> edge_j_comb(num_threads);
    #pragma omp parallel for
    for (int t = 0; t < num_threads; t++)
    {
        const int num_th = omp_get_thread_num();
        std::vector<int> edge_i_thread, edge_j_thread;
        std::tie(edge_i_thread, edge_j_thread) = compute_incorrect_edge_indices_helper(node_labels_a, node_labels_b, num_sampled_nodes / num_threads, num_sampled_nodes);
        edge_i_comb[t] = edge_i_thread;
        edge_j_comb[t] = edge_j_thread;
    }
    int numel = 0;
    std::vector<int> cumm_sizes;
    cumm_sizes.push_back(0);
    for (int t = 0; t < num_threads; t++)
    {
        cumm_sizes.push_back(numel + edge_i_comb[t].size());
        numel += edge_i_comb[t].size();
    }
    
    std::vector<int> edge_i_full(numel, 0);
    std::vector<int> edge_j_full(numel, 0);
    #pragma omp parallel for
    for (int t = 0; t < num_threads; t++)
    {
        std::copy(edge_i_comb[t].begin(), edge_i_comb[t].end(), edge_i_full.begin() + cumm_sizes[t]);
        std::copy(edge_i_comb[t].begin(), edge_i_comb[t].end(), edge_j_full.begin() + cumm_sizes[t]);
    }
    return {edge_i_full, edge_j_full};
}

PYBIND11_MODULE(rand_index_py, m) {
    m.doc() = "Utils for computing Rand Index efficiently.";
    m.def("compute_incorrect_edge_indices", [](const std::vector<int>& node_labels_a, const std::vector<int>& node_labels_b) {
            return compute_incorrect_edge_indices(node_labels_a, node_labels_b);
            });
    m.def("compute_incorrect_edge_indices_sampled", [](const std::vector<int>& node_labels_a, const std::vector<int>& node_labels_b, const float subsampling_factor) {
        return compute_incorrect_edge_indices_sampled(node_labels_a, node_labels_b, subsampling_factor);
        });
}

