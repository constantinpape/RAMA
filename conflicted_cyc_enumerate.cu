#include "conflicted_cyc_enumerate.h"
#include "utils.h"

#define tol 1e-6

typedef std::vector<thrust::device_vector<int>> device_vectors;
struct is_positive_edge
{
    __host__ __device__ bool operator()(const thrust::tuple<int,int,float>& t)
    {
        return thrust::get<2>(t) > tol;
    }
};

struct is_neg_edge
{
    __host__ __device__ bool operator()(const thrust::tuple<int,int,float>& t)
    {
        return thrust::get<2>(t) < tol;
    }
};

std::tuple<dCOO, thrust::device_vector<int>, thrust::device_vector<int>> create_matrices(const dCOO& A)
{
    MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME
    
    // Partition edges into positive and negative.
    thrust::device_vector<int> row_ids = A.get_row_ids();
    thrust::device_vector<int> col_ids = A.get_col_ids();
    thrust::device_vector<float> costs = A.get_data();

    auto first = thrust::make_zip_iterator(thrust::make_tuple(row_ids.begin(), col_ids.begin(), costs.begin()));
    auto last = thrust::make_zip_iterator(thrust::make_tuple(row_ids.end(), col_ids.end(), costs.end()));

    thrust::device_vector<int> row_ids_pos(row_ids.size());
    thrust::device_vector<int> col_ids_pos(row_ids.size());
    thrust::device_vector<float> costs_pos(row_ids.size());
    auto first_pos = thrust::make_zip_iterator(thrust::make_tuple(row_ids_pos.begin(), col_ids_pos.begin(), costs_pos.begin()));
    auto last_pos = thrust::copy_if(first, last, first_pos, is_positive_edge());
    const size_t nr_positive_edges = std::distance(first_pos, last_pos);
    row_ids_pos.resize(nr_positive_edges);
    col_ids_pos.resize(nr_positive_edges);
    costs_pos.resize(nr_positive_edges);

    thrust::device_vector<int> row_ids_neg(row_ids.size());
    thrust::device_vector<int> col_ids_neg(row_ids.size());
    thrust::device_vector<float> costs_neg(row_ids.size());
    auto first_neg = thrust::make_zip_iterator(thrust::make_tuple(row_ids_neg.begin(), col_ids_neg.begin(), costs_neg.begin()));
    auto last_neg = thrust::copy_if(first, last, first_neg, is_neg_edge());
    const size_t nr_neg_edges = std::distance(first_neg, last_neg);
    row_ids_neg.resize(nr_neg_edges);
    col_ids_neg.resize(nr_neg_edges);

    // Create symmetric adjacency matrix of positive edges.
    dCOO A_pos_symm;
    if (nr_positive_edges > 0)
    {
        std::tie(row_ids_pos, col_ids_pos, costs_pos) = to_undirected(row_ids_pos, col_ids_pos, costs_pos);
        A_pos_symm = dCOO(std::max(A.rows(), A.cols()), std::max(A.rows(), A.cols()),
                    std::move(col_ids_pos), std::move(row_ids_pos), std::move(costs_pos));
    }
    return {A_pos_symm, row_ids_neg, col_ids_neg};
}

//TODO: Try replacing with gather, scatter ops.
__global__ void copy_neighbourhood(const int num_edges,
                                const int* const __restrict__ v_frontier_counts,
                                const int* const __restrict__ v_frontier_offsets,
                                const int* const __restrict__ expanded_src_compressed,
                                const int* const __restrict__ expanded_dst,
                                const int* const __restrict__ v_frontier_parent_edge,
                                const int* const __restrict__ v_frontier_all_src_offsets,
                                int* __restrict__ v_frontier_all_neighbours,
                                int* __restrict__ v_frontier_all_parent_edge)
{
    const int start_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_threads = blockDim.x * gridDim.x;

    for (int e = start_index; e < num_edges; e += num_threads) 
    {
        const int src_compressed = expanded_src_compressed[e];
        const int input_count = v_frontier_counts[src_compressed];
        int output_start_offset = v_frontier_all_src_offsets[src_compressed];
        const int output_end_offset = v_frontier_all_src_offsets[src_compressed + 1];
        const int num_neighbours = (output_end_offset - output_start_offset) / input_count;
        printf("e: %d, src: %d, input_count: %d, output_start_offset: %d, output_end_offset: %d, num_neighbours: %d \n", 
                e, src_compressed, input_count, output_start_offset, output_end_offset, num_neighbours);
        if (num_neighbours == 0)
            continue;
        const int v_frontier_parent_start_index = v_frontier_offsets[src_compressed];
        const int dst = expanded_dst[e];
        for (int c = 0; c != input_count; ++c, output_start_offset += num_neighbours)
        {
            const int parent_edge = v_frontier_parent_edge[v_frontier_parent_start_index];
            v_frontier_all_neighbours[output_start_offset] = dst;
            v_frontier_all_parent_edge[output_start_offset] = parent_edge;
        }
    }
}

thrust::device_vector<int> invert_unique(const thrust::device_vector<int>& values, const thrust::device_vector<int>& counts)
{
    thrust::device_vector<int> counts_sum(counts.size());
    thrust::inclusive_scan(counts.begin(), counts.end(), counts_sum.begin());
    
    int out_size = counts_sum[counts_sum.size() - 1];
    thrust::device_vector<int> output_indices(out_size);

    thrust::scatter(thrust::make_constant_iterator(1), thrust::make_constant_iterator(1), counts_sum.begin(), output_indices.begin());
    thrust::inclusive_scan(output_indices.begin(), output_indices.end(), output_indices.begin());

    thrust::device_vector<int> out_values(out_size);
    thrust::gather(output_indices.begin(), output_indices.end(), values.begin(), out_values.begin());
    return out_values;
}

std::tuple<thrust::device_vector<int>, thrust::device_vector<int>> get_unique_with_counts(const thrust::device_vector<int>& input)
{
    assert(thrust::is_sorted(input.begin(), input.end()));
    thrust::device_vector<int> unique_counts(input.size() + 1);
    thrust::device_vector<int> unique_values(input.size());

    auto new_end = thrust::unique_by_key_copy(input.begin(), input.end(), thrust::make_counting_iterator(0), unique_values.begin(), unique_counts.begin());
    int num_unique = std::distance(unique_values.begin(), new_end.first);
    unique_values.resize(num_unique);
    unique_counts.resize(num_unique + 1); // contains smallest index of each unique element.
    
    unique_counts[num_unique] = input.size();
    thrust::adjacent_difference(unique_counts.begin(), unique_counts.end(), unique_counts.begin());
    unique_counts = thrust::device_vector<int>(unique_counts.begin() + 1, unique_counts.end());

    return {unique_values, unique_counts};
}

struct edge_in_frontier
{
    const int* v_frontier_indicator;
    __host__ __device__ bool operator()(const thrust::tuple<int,int>& t)
    {
        int i = thrust::get<0>(t);
        if(v_frontier_indicator[i] != -1)
            return true;
        return false;
    }
};

struct map_src_nodes
{
    const int* v_frontier_indicator;
    __host__ __device__ int operator()(const int& v)
    {
        return v_frontier_indicator[v];
    }
};

std::tuple<thrust::device_vector<int>, thrust::device_vector<int>, thrust::device_vector<int>, thrust::device_vector<int>> 
    expand_frontier(const thrust::device_vector<int>& v_parent_edges, const thrust::device_vector<int>& v_frontier, const dCOO& A_pos)
{
    assert(v_parent_edges.size() == v_frontier.size());
    assert(thrust::is_sorted(v_frontier.begin(), v_frontier.end())); // can contain duplicates.
    // 1. Find unique vertices and their counts:
    thrust::device_vector<int> v_frontier_unique, v_frontier_counts;
    thrust::copy(v_frontier.begin(), v_frontier.end(), std::ostream_iterator<int>(std::cout, " "));
    std::cout<<"\n";
    std::tie(v_frontier_unique, v_frontier_counts) = get_unique_with_counts(v_frontier);
    thrust::copy(v_frontier_unique.begin(), v_frontier_unique.end(), std::ostream_iterator<int>(std::cout, " "));
    std::cout<<"\n";
    thrust::copy(v_frontier_counts.begin(), v_frontier_counts.end(), std::ostream_iterator<int>(std::cout, " "));
    std::cout<<"\n";

    // 2. Iterate over all edges and keep the ones whose first end-point is in the frontier. 
    thrust::device_vector<int> v_frontier_indicator(std::max(A_pos.rows(), A_pos.cols()), -1);
    thrust::scatter(thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(v_frontier_unique.size()), v_frontier_unique.begin(), v_frontier_indicator.begin());
    thrust::device_vector<int> row_ids = A_pos.get_row_ids();
    thrust::device_vector<int> col_ids = A_pos.get_col_ids();
    auto first = thrust::make_zip_iterator(thrust::make_tuple(row_ids.begin(), col_ids.begin()));
    auto last = thrust::make_zip_iterator(thrust::make_tuple(row_ids.end(), col_ids.end()));
    thrust::device_vector<int> expanded_src(A_pos.edges());
    thrust::device_vector<int> expanded_dst(A_pos.edges());
    auto first_output = thrust::make_zip_iterator(thrust::make_tuple(expanded_src.begin(), expanded_dst.begin()));
    edge_in_frontier expansion_func({thrust::raw_pointer_cast(v_frontier_indicator.data())}); 
    auto last_output = thrust::copy_if(first, last, first_output, expansion_func);
    int num_expanded_edges = std::distance(first_output, last_output);
    expanded_src.resize(num_expanded_edges);
    expanded_dst.resize(num_expanded_edges);

    thrust::copy(expanded_src.begin(), expanded_src.end(), std::ostream_iterator<int>(std::cout, " "));
    std::cout<<"\n";

    thrust::copy(expanded_dst.begin(), expanded_dst.end(), std::ostream_iterator<int>(std::cout, " "));
    std::cout<<"\n";

    std::cout<<num_expanded_edges<<"\n";

    // coo_sorting(expanded_dst, expanded_src); // should already be sorted.
    // 3. Check how many neighbours of each unique frontier vertex are created:
    thrust::device_vector<int> expanded_src_unique, expanded_src_count;
    std::tie(expanded_src_unique, expanded_src_count) = get_unique_with_counts(expanded_src);
    assert(thrust::equal(expanded_src_unique.begin(), expanded_src_unique.end(), v_frontier_unique.begin()));

    // 4. Now revert step 1. and account for duplicates:
    thrust::device_vector<int> out_src_offsets(v_frontier_counts.size() + 1); // is calculated w.r.t unique vertices in frontier!
    thrust::transform(v_frontier_counts.begin(), v_frontier_counts.end(), expanded_src_count.begin(), out_src_offsets.begin(), thrust::multiplies<int>());
    thrust::fill(out_src_offsets.begin() + out_src_offsets.size(), out_src_offsets.end(), 0);
    thrust::exclusive_scan(out_src_offsets.begin(), out_src_offsets.end(), out_src_offsets.begin());
    int output_num = out_src_offsets.back() + 1;
    thrust::device_vector<int> expanded_dst_all(output_num, -1); //TODO: remove -1.
    thrust::device_vector<int> expanded_dst_parent_edges(output_num, -1);
    // compress source vertex labels:
    map_src_nodes src_compress({thrust::raw_pointer_cast(v_frontier_indicator.data())}); 
    thrust::transform(expanded_src.begin(), expanded_src.end(), expanded_src.begin(), src_compress);

    thrust::device_vector<int> v_frontier_offsets(v_frontier_counts.size() + 1);
    thrust::exclusive_scan(v_frontier_counts.begin(), v_frontier_counts.end(), v_frontier_offsets.begin());
    thrust::fill(v_frontier_offsets.begin() + v_frontier_counts.size(), v_frontier_offsets.end(), v_frontier_counts[v_frontier_counts.size() - 1] + v_frontier_offsets[v_frontier_counts.size() - 1]);

    thrust::copy(v_frontier_counts.begin(), v_frontier_counts.end(), std::ostream_iterator<int>(std::cout, " "));
    std::cout<<"\n";

    thrust::copy(out_src_offsets.begin(), out_src_offsets.end(), std::ostream_iterator<int>(std::cout, " "));
    std::cout<<"\n";

    thrust::copy(expanded_src.begin(), expanded_src.end(), std::ostream_iterator<int>(std::cout, " "));
    std::cout<<"\n";

    thrust::copy(v_frontier_offsets.begin(), v_frontier_offsets.end(), std::ostream_iterator<int>(std::cout, " "));
    std::cout<<"\n";

    thrust::copy(v_parent_edges.begin(), v_parent_edges.end(), std::ostream_iterator<int>(std::cout, " "));
    std::cout<<"\n";

    int threadCount = 256;
    int blockCount = ceil(num_expanded_edges / (float) threadCount);
    copy_neighbourhood<<<blockCount, threadCount>>>(num_expanded_edges, 
                        thrust::raw_pointer_cast(v_frontier_counts.data()),
                        thrust::raw_pointer_cast(v_frontier_offsets.data()),
                        thrust::raw_pointer_cast(expanded_src.data()),
                        thrust::raw_pointer_cast(expanded_dst.data()),
                        thrust::raw_pointer_cast(v_parent_edges.data()),
                        thrust::raw_pointer_cast(out_src_offsets.data()),
                        thrust::raw_pointer_cast(expanded_dst_all.data()),
                        thrust::raw_pointer_cast(expanded_dst_parent_edges.data()));

    thrust::device_vector<int> v_frontier_num_neighbours = invert_unique(expanded_src_count, v_frontier_counts);
    thrust::copy(expanded_dst_all.begin(), expanded_dst_all.end(), std::ostream_iterator<int>(std::cout, " "));
    std::cout<<"\n";

    thrust::copy(expanded_dst_parent_edges.begin(), expanded_dst_parent_edges.end(), std::ostream_iterator<int>(std::cout, " "));
    std::cout<<"\n";

    return {v_frontier_num_neighbours, out_src_offsets, expanded_dst_all, expanded_dst_parent_edges};

    // thrust::device_vector<int> A_pos_degrees(A_pos_row_offsets.size());
    // thrust::adjacent_difference(A_pos_row_offsets.begin(), A_pos_row_offsets.end(), A_pos_degrees.begin()); // degree for node n is at n + 1 location. 

    // thrust::device_vector<int> v_frontier_row_offsets(v_frontier.size() + 1);
    // v_frontier_row_offsets[0] = 0;
    // thrust::gather(v_frontier.begin(), v_frontier.end(), A_pos_degrees.begin() +  1, v_frontier_row_offsets.begin() + 1); // gather the nodes in frontier.
    // thrust::inclusive_scan(v_frontier_row_offsets.begin(), v_frontier_row_offsets.end(), v_frontier_row_offsets.begin()); // convert from degrees to offsets.

    // thrust::device_vector<int> v_expanded(v_frontier_row_offsets[v_frontier_row_offsets.size() - 1]);
    // thrust::device_vector<int> v_frontier_valid_node_degrees(v_frontier.size() + 1, 0);

    // return edge list of {v_frontier, v_expanded}. COO or not?
}

struct vertex_and_parent_compare
{
    __host__ __device__ bool operator()(const thrust::tuple<int,int>& t1, const thrust::tuple<int,int>& t2)
    {
        const int v1 = thrust::get<0>(t1);
        const int v2 = thrust::get<0>(t2);
        if (v1 != v2)
            return v1 < v2;

        const int p1 = thrust::get<1>(t1);
        const int p2 = thrust::get<1>(t2);
        return p1 < p2;
    }
};

std::tuple<thrust::device_vector<int>, thrust::device_vector<int>, thrust::device_vector<int>> merge_paths(
        const device_vectors& up_num_neigh, const device_vectors& up_offsets, const device_vectors& up_dst, const device_vectors& up_parents,
        const device_vectors& down_num_neigh, const device_vectors& down_offsets, const device_vectors& down_dst, const device_vectors& down_parents,
        const thrust::device_vector<int>& row_ids_rep, const thrust::device_vector<int>& col_ids_rep)
{
    assert(up_dst.size() == up_parents.size());
    assert(down_dst.size() == down_parents.size());

    // For both up and down paths, create a container with (vertex, parent edge id) and sort them.
    thrust::device_vector<int> up_final_dst = up_dst[up_dst.size() - 1];
    std::cout<< up_final_dst.size() <<"\n";
    thrust::device_vector<int> up_final_parents = up_parents[up_parents.size() - 1];
    auto first_up = thrust::make_zip_iterator(thrust::make_tuple(up_final_dst.begin(), up_final_parents.begin()));
    auto last_up = thrust::make_zip_iterator(thrust::make_tuple(up_final_dst.end(), up_final_parents.end()));
    thrust::sort(first_up, last_up, vertex_and_parent_compare());

    thrust::device_vector<int> down_final_dst = down_dst[down_dst.size() - 1];
    std::cout<< down_final_dst.size() <<"\n";
    thrust::device_vector<int> down_final_parents = down_parents[down_parents.size() - 1];
    auto first_down = thrust::make_zip_iterator(thrust::make_tuple(down_final_dst.begin(), down_final_parents.begin()));
    auto last_down = thrust::make_zip_iterator(thrust::make_tuple(down_final_dst.end(), down_final_parents.end()));
    thrust::sort(first_down, last_down, vertex_and_parent_compare());

    // now find intersecting vertices and edges.
    thrust::device_vector<int> intersect_dst(up_dst.size() + down_dst.size());
    thrust::device_vector<int> intersect_parents(up_dst.size() + down_dst.size());
    auto first_intersect = thrust::make_zip_iterator(thrust::make_tuple(intersect_dst.begin(), intersect_parents.begin()));
    auto last_intersect = thrust::set_intersection(first_up, last_up, first_down, last_down, first_intersect, vertex_and_parent_compare());
    int number_intersect = std::distance(first_intersect, last_intersect);

    std::cout<< number_intersect <<"\n";
    intersect_dst.resize(number_intersect);
    intersect_parents.resize(number_intersect);

    thrust::device_vector<int> tri_v1(number_intersect);
    thrust::device_vector<int> tri_v2(number_intersect);
    thrust::device_vector<int> tri_v3 = intersect_dst;

    thrust::gather(intersect_parents.begin(), intersect_parents.end(), row_ids_rep.begin(), tri_v1.begin());
    thrust::gather(intersect_parents.begin(), intersect_parents.end(), col_ids_rep.begin(), tri_v2.begin());
    return {tri_v1, tri_v2, tri_v3};
}

// A should be directed thus containing same number of elements as in original problem.
std::tuple<thrust::device_vector<int>, thrust::device_vector<int>, thrust::device_vector<int>> enumerate_conflicted_cycles(const dCOO& A, const int max_cycle_length)
{
    MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME;
    dCOO A_pos_symm;
    thrust::device_vector<int> row_ids_rep, col_ids_rep;
    std::tie(A_pos_symm, row_ids_rep, col_ids_rep) = create_matrices(A); // TODO: Preprocess and remove edges from 'A_pos_symm' which are more than 'max_cycle_length' away from any repulsive edge?

    // Initialize:
    thrust::device_vector<int> edge_ids(row_ids_rep.size());
    thrust::sequence(edge_ids.begin(), edge_ids.end());

    device_vectors up_v_frontier_num_neigh, down_v_frontier_num_neigh;
    device_vectors up_src_offsets, down_src_offsets;
    device_vectors up_expanded, down_expanded;
    device_vectors up_expanded_parent_edges, down_expanded_parent_edges;

    // expand rows:
    thrust::device_vector<int> row_seeds = row_ids_rep;
    thrust::device_vector<int> edge_ids_rows = edge_ids;
    thrust::sort_by_key(row_seeds.begin(), row_seeds.end(), edge_ids_rows.begin());
    thrust::device_vector<int> v_frontier_num_neigh, out_src_offsets, expanded_dst_all, expanded_dst_parent_edges;
    std::tie(v_frontier_num_neigh, out_src_offsets, expanded_dst_all, expanded_dst_parent_edges) = expand_frontier(edge_ids_rows, row_seeds, A_pos_symm);
    up_v_frontier_num_neigh.push_back(v_frontier_num_neigh);
    up_src_offsets.push_back(out_src_offsets);
    up_expanded.push_back(expanded_dst_all);
    up_expanded_parent_edges.push_back(expanded_dst_parent_edges);

    // expand cols:
    thrust::device_vector<int> col_seeds = col_ids_rep;
    thrust::device_vector<int> edge_ids_cols = edge_ids;
    thrust::sort_by_key(col_seeds.begin(), col_seeds.end(), edge_ids_cols.begin());
    std::tie(v_frontier_num_neigh, out_src_offsets, expanded_dst_all, expanded_dst_parent_edges) = expand_frontier(edge_ids_cols, col_seeds, A_pos_symm);

    down_v_frontier_num_neigh.push_back(v_frontier_num_neigh);
    down_src_offsets.push_back(out_src_offsets);
    down_expanded.push_back(expanded_dst_all);
    down_expanded_parent_edges.push_back(expanded_dst_parent_edges);

    thrust::device_vector<int> tri_v1, tri_v2, tri_v3;
    std::tie(tri_v1, tri_v2, tri_v3) = merge_paths(up_v_frontier_num_neigh, up_src_offsets, up_expanded, up_expanded_parent_edges,
                                                down_v_frontier_num_neigh, down_src_offsets, down_expanded, down_expanded_parent_edges,
                                                row_ids_rep, col_ids_rep);
    // for (int i = 0; i < max_cycle_length - 2; i++)
    // {

    // }
    return {tri_v1, tri_v2, tri_v3};
}