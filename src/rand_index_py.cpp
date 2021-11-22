#include <vector>
#include <tuple>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py=pybind11;

std::tuple<std::vector<int>, std::vector<int>> compute_incorrect_edge_indices(
    const std::vector<int>& node_labels_a,
    const std::vector<int>& node_labels_b) 
{
    const int num_nodes = node_labels_a.size();
    std::vector<int> edge_i, edge_j;
    edge_i.reserve(num_nodes);
    edge_j.reserve(num_nodes);
    for (int i = 0; i != num_nodes; ++i) //TODO use GPU?
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

PYBIND11_MODULE(rand_index_py, m) {
    m.doc() = "Utils for computing Rand Index efficiently.";
    m.def("compute_incorrect_edge_indices", [](const std::vector<int>& node_labels_a, const std::vector<int>& node_labels_b) {
            return compute_incorrect_edge_indices(node_labels_a, node_labels_b);
            });
}

