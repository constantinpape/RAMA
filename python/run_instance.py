import pickle, os
from parallel_gaec_py import parallel_gaec_eigen
from parallel_gaec_torch import parallel_gaec_torch
# from lpmp_py.raw_solvers import amwc_solver # For comparison with LPMP GAEC.
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt 
import time 

def get_mask_image(cluster_labels_1d):
    # Does not create good visualizations if the number of clusters are high due to overflowing colormap.
    assert len(cluster_labels_1d) == HEIGHT * WIDTH
    cluster_labels_2d = np.array(cluster_labels_1d).reshape(HEIGHT, WIDTH)

    mask_image = np.zeros(cluster_labels_2d.shape, dtype=np.float32)
    values = np.unique(cluster_labels_2d)
    random_labels = np.random.randint(0, len(values), len(values))
    for (i, v) in enumerate(values):
        mask_image[cluster_labels_2d == v] = random_labels[i]
    print(f"Found: {i + 1} clusters.")
    mask_image = mask_image / i
    mask_image = plt.cm.tab20(mask_image)
    mask_image = (mask_image * 255).astype(np.uint8)
    return Image.fromarray(mask_image)

def get_edge_image(cluster_labels_1d):
    assert len(cluster_labels_1d) == HEIGHT * WIDTH
    cluster_labels_2d = np.array(cluster_labels_1d).reshape(HEIGHT, WIDTH)

    edge_image = np.zeros(cluster_labels_2d.shape, dtype=np.float32)
    edge_image[:, :-1] += cluster_labels_2d[:, 1:] != cluster_labels_2d[:, :-1]
    edge_image[:-1, :] += cluster_labels_2d[1:, :] != cluster_labels_2d[:-1, :]
    edge_image[edge_image > 1] = 1
    return Image.fromarray((edge_image * 255).astype(np.uint8))

def compute_cost(edge_indices, edge_costs, node_labels):
    node_labels = np.array(node_labels)
    row = edge_indices[:, 0]; col = edge_indices[:, 1]
    row_label = node_labels[row]; col_label = node_labels[col]
    edge_labels = row_label != col_label
    return (edge_labels * edge_costs).sum()

# Contains many instances of small MC problems:
instance_dir = '/BS/ahmed_projects/work/data/multicut/cityscapes_small_val_instances/'; HEIGHT = 256; WIDTH = 512

# Contains two instances of large MC problems:
# instance_dir = "/BS/ahmed_projects/work/data/multicut/cityscapes_large_instances/"; HEIGHT = 1024; WIDTH = 2048

# Directory to save results (should exist before):
results_folder = 'out/.'
number_to_run = 0
for (i, file_name) in enumerate(sorted(os.listdir(instance_dir))):
    file_path = os.path.join(instance_dir, file_name)
    file_name_prefix = os.path.splitext(file_name)[0]
    if not file_path.endswith('.pkl'):
        continue
    print(f"Loading instance {file_path}...")
    instance = pickle.load(open(file_path, 'rb'))
    edge_indices = instance['edge_indices']; edge_costs = instance['edge_costs']

    print("Running parallel GAEC Python...")
    start = time.time()
    labels_parallel_gaec_torch = parallel_gaec_torch(edge_indices.transpose(), edge_costs.squeeze(), 0.05, 1)
    end = time.time()
    print(f"Parallel GAEC torch time: {end - start:.3f}secs")
    labels_img = get_mask_image(labels_parallel_gaec_torch)
    labels_img.save(os.path.join(results_folder, file_name_prefix + 'parallel_gaec_result_torch.png'))

    edge_image = get_edge_image(labels_parallel_gaec_torch)
    edge_image.save(os.path.join(results_folder, file_name_prefix + 'parallel_gaec_edge_image_torch.png'))


    print("Running parallel GAEC...")
    start = time.time()
    labels_parallel_gaec = parallel_gaec_eigen(edge_indices, edge_costs, 0.05, 1)
    end = time.time()
    print(f"Parallel GAEC time: {end - start:.3f}secs")
    labels_img = get_mask_image(labels_parallel_gaec)
    labels_img.save(os.path.join(results_folder, file_name_prefix + 'parallel_gaec_result.png'))

    edge_image = get_edge_image(labels_parallel_gaec)
    edge_image.save(os.path.join(results_folder, file_name_prefix + 'parallel_gaec_edge_image.png'))

    # For running AMWC from LPMP: (https://github.com/aabbas90/LPMP/tree/268132a4b95308ae1e8a1afeadeac1da2e63e8f9) 
    # num_nodes = instance['edge_indices'].max() + 1
    # node_costs = np.zeros((num_nodes, 1)) # Create fake node costs.
    # start = time.time()
    # _, labels_seq_gaec, _, solver_cost = amwc_solver(node_costs, edge_indices, edge_costs, partitionable = np.array([True] * 1, dtype='bool'))
    # end = time.time()
    # seq_gaec_time = end - start
    # print(f"Seq. GAEC time: {seq_gaec_time:.3f}")
    # labels_img = get_mask_image(labels_seq_gaec)
    # labels_img.save(os.path.join(results_folder, file_name_prefix + '_seq_gaec_result.png'))

    # edge_image = get_edge_image(labels_seq_gaec)
    # edge_image.save(os.path.join(results_folder, file_name_prefix + '_seq_gaec_edge_image.png'))

    if i >= number_to_run:
        break