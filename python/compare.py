import pickle, os
from parallel_gaec_py import parallel_gaec_eigen
from parallel_gaec_torch import parallel_gaec_torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt 
import time 

def compute_cost(edge_indices, edge_costs, node_labels):
    node_labels = np.array(node_labels)
    row = edge_indices[:, 0]; col = edge_indices[:, 1]
    row_label = node_labels[row]; col_label = node_labels[col]
    edge_labels = row_label != col_label
    return (edge_labels * edge_costs).sum()

edge_indices = np.array([[0, 1], [0, 2], [1, 2], [2, 3], [0, 3]])
edge_costs = np.array([0.5, -3.0, 1.0, 1.0, 1.0], dtype = np.float32)

labels_parallel_gaec_torch = parallel_gaec_torch(edge_indices.transpose(), edge_costs.squeeze(), 0.001, 10)
print(labels_parallel_gaec_torch)
print(compute_cost(edge_indices, edge_costs, labels_parallel_gaec_torch))

labels_parallel_gaec_cpp = parallel_gaec_eigen(edge_indices, edge_costs, 0.001, 10)
print(labels_parallel_gaec_cpp)
print(compute_cost(edge_indices, edge_costs, labels_parallel_gaec_cpp))