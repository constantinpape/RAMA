#pragma once

#include <cassert>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/extrema.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>
#include <thrust/generate.h>
#include <cusparse.h>
#include "time_measure_util.h"

class dEdgeList {
    public:
        dEdgeList() {}

        template<typename COL_ITERATOR, typename ROW_ITERATOR, typename DATA_ITERATOR>
            dEdgeList(cusparseHandle_t handle, 
                    const int _rows, const int _cols,
                    COL_ITERATOR col_id_begin, COL_ITERATOR col_id_end,
                    ROW_ITERATOR row_id_begin, ROW_ITERATOR row_id_end,
                    DATA_ITERATOR data_begin, DATA_ITERATOR data_end);

        template<typename COL_ITERATOR, typename ROW_ITERATOR, typename DATA_ITERATOR>
            dEdgeList(cusparseHandle_t handle, 
                    COL_ITERATOR col_id_begin, COL_ITERATOR col_id_end,
                    ROW_ITERATOR row_id_begin, ROW_ITERATOR row_id_end,
                    DATA_ITERATOR data_begin, DATA_ITERATOR data_end);

        size_t rows() const { return rows_; }
        size_t cols() const { return cols_; }
        size_t edges() const { return row_ids.size(); }
        float sum();

        void remove_diagonal(cusparseHandle_t handle);

        const thrust::device_vector<int> get_row_ids() const { return row_ids; }
        const thrust::device_vector<int> get_col_ids() const { return col_ids; }
        const thrust::device_vector<float> get_data() const { return data; }

        thrust::device_vector<float> diagonal(cusparseHandle_t) const;
        dEdgeList contract_cuda(cusparseHandle_t handle, const thrust::device_vector<int>& node_mapping);

    private:
        template<typename COL_ITERATOR, typename ROW_ITERATOR, typename DATA_ITERATOR>
            void init(cusparseHandle_t handle,
                    COL_ITERATOR col_id_begin, COL_ITERATOR col_id_end,
                    ROW_ITERATOR row_id_begin, ROW_ITERATOR row_id_end,
                    DATA_ITERATOR data_begin, DATA_ITERATOR data_end);
        int rows_ = 0;
        int cols_ = 0;
        thrust::device_vector<float> data;
        thrust::device_vector<int> row_ids;
        thrust::device_vector<int> col_ids; 
};

    template<typename COL_ITERATOR, typename ROW_ITERATOR, typename DATA_ITERATOR>
dEdgeList::dEdgeList(cusparseHandle_t handle, 
        const int _rows, const int _cols,
        COL_ITERATOR col_id_begin, COL_ITERATOR col_id_end,
        ROW_ITERATOR row_id_begin, ROW_ITERATOR row_id_end,
        DATA_ITERATOR data_begin, DATA_ITERATOR data_end)
    : rows_(_rows),
    cols_(_cols)
{
    init(handle, col_id_begin, col_id_end, row_id_begin, row_id_end, data_begin, data_end);
} 

    template<typename COL_ITERATOR, typename ROW_ITERATOR, typename DATA_ITERATOR>
dEdgeList::dEdgeList(cusparseHandle_t handle,
        COL_ITERATOR col_id_begin, COL_ITERATOR col_id_end,
        ROW_ITERATOR row_id_begin, ROW_ITERATOR row_id_end,
        DATA_ITERATOR data_begin, DATA_ITERATOR data_end)
{
    init(handle, col_id_begin, col_id_end, row_id_begin, row_id_end, data_begin, data_end);
}

    template<typename COL_ITERATOR, typename ROW_ITERATOR, typename DATA_ITERATOR>
void dEdgeList::init(cusparseHandle_t handle,
        COL_ITERATOR col_id_begin, COL_ITERATOR col_id_end,
        ROW_ITERATOR row_id_begin, ROW_ITERATOR row_id_end,
        DATA_ITERATOR data_begin, DATA_ITERATOR data_end)
{
    assert(std::distance(data_begin, data_end) == std::distance(col_id_begin, col_id_end));
    assert(std::distance(data_begin, data_end) == std::distance(row_id_begin, row_id_end));

    std::cout << "Allocation matrix with " << std::distance(data_begin, data_end) << " entries\n";
    row_ids = thrust::device_vector<int>(row_id_begin, row_id_end);
    col_ids = thrust::device_vector<int>(col_id_begin, col_id_end);
    data = thrust::device_vector<float>(data_begin, data_end);

    if(cols_ == 0)
        cols_ = *thrust::max_element(col_ids.begin(), col_ids.end()) + 1;
    assert(cols_ > *thrust::max_element(col_ids.begin(), col_ids.end()));
    if(rows_ == 0)
        rows_ = *thrust::max_element(row_ids.begin(), row_ids.end()) + 1;
    assert(rows_ > *thrust::max_element(row_ids.begin(), row_ids.end()));
}