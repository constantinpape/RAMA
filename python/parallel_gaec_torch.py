from numpy.core.fromnumeric import size
import torch 
from networkx.utils import UnionFind
import time
import logging
logging.basicConfig(level=logging.INFO, format='%(message)s', datefmt='%S')
# create a logging format

def initialize_adjacency(num_nodes, edge_indices, edge_costs):
    assert edge_indices.shape[0] == 2
    row, col = edge_indices
    row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
    edge_indices = torch.stack([row, col], dim=0)
    edge_costs = torch.cat([edge_costs, edge_costs], dim=0)
    return torch.sparse_coo_tensor(indices = edge_indices, 
                                values = edge_costs,
                                size = (num_nodes, num_nodes),
                                device = edge_costs.device).coalesce()


def contraction_matrix(num_nodes, A, fraction):
    edge_indices = A.indices()
    edge_costs = A.values()
    upper_triangle = edge_indices[1, :] > edge_indices[0, :]
    edge_indices = edge_indices[:, upper_triangle]
    edge_costs = edge_costs[upper_triangle]
    device = edge_costs.device

    #num_elements = int(fraction * len(edge_costs)) + 1
    num_elements = int(fraction * num_nodes) + 1 # TODO: This relation matches with cpp implementation but only for first iteration.
    top_values, top_indices = torch.topk(edge_costs, k = num_elements, sorted=False)
    positive_mask = top_values > 0
    top_pos_indices = top_indices[positive_mask]
    row_indices = torch.cat((torch.arange(0, num_nodes, device=device), edge_indices[0, top_pos_indices]), 0)
    col_indices = torch.cat((torch.arange(0, num_nodes, device=device), edge_indices[1, top_pos_indices]), 0)
    E = torch.sparse_coo_tensor(indices = torch.stack((row_indices, col_indices), 0), 
                                values = torch.ones_like(row_indices, dtype=torch.float32),
                                size = (num_nodes, num_nodes),
                                device = device).coalesce()
    return E, edge_indices[0, top_pos_indices], edge_indices[1, top_pos_indices] 

def eliminate_zeros(x, nr, nc):
    mask = x._values().nonzero()
    nv = x._values().index_select(0, mask.view(-1))
    ni = x._indices().index_select(1, mask.view(-1))
    return torch.sparse.FloatTensor(ni, nv, torch.Size([nr, nc]))

def remove_nodes_from_adj(num_nodes, nodes_to_remove, A):
    ind = torch.arange(num_nodes, device=A.device)
    val = torch.ones_like(ind, dtype=torch.float32)
    val[nodes_to_remove] = 0.0
    valid_mask = val > 0
    R = torch.sparse_coo_tensor(indices = torch.stack((ind[valid_mask], ind[valid_mask]), 0), values = val[valid_mask], size = (num_nodes, num_nodes))
    return eliminate_zeros(torch.sparse.mm(torch.sparse.mm(R, A), R.transpose(1, 0)).coalesce(), num_nodes, num_nodes)

def parallel_gaec_torch(edge_indices, edge_costs, fraction, itr):
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.DEBUG)

    assert len(edge_costs.shape) == 1, f"edge_costs of shape: {edge_costs.shape} should be a vector of length M."
    assert edge_indices.shape[0] == 2, f"edge_indices of shape: {edge_indices.shape} should be 2 X M."
    assert edge_indices.shape[1] == edge_costs.shape[0], f"edge_indices ({edge_indices.shape[1]}), costs({edge_costs.shape[0]}) should have same length."

    num_nodes = edge_indices.max() + 1
    uf = UnionFind(elements = [n for n in range(num_nodes)])
    edge_costs = torch.from_numpy(edge_costs).to('cuda')    
    edge_indices = torch.from_numpy(edge_indices).to('cuda')    
    A = initialize_adjacency(num_nodes, edge_indices, edge_costs)
    for i in range(itr):

        start = time.time()
        # 1. Create contraction matrix:
        E, row, col = contraction_matrix(num_nodes, A, fraction)
        if row.numel() == 0:
            break
        logger.debug(f"Spent: {time.time() - start:.3f}s in itr: {i}, contraction_matrix")

        # 2. Left and right multiply with A to contract edge (i, j) 
        # towards node i where i < j (Since E is upper triangular.)
        start = time.time()
        A = torch.sparse.mm(torch.sparse.mm(E, A), E.transpose(1, 0)).coalesce()
        logger.debug(f"Spent: {time.time() - start:.3f}s  in itr: {i}, mm")

        # 3. Remove node j from adjacency (both row, col) and also set diag to zero:
        start = time.time()
        A = remove_nodes_from_adj(num_nodes, col, A).coalesce()
        logger.debug(f"Spent: {time.time() - start:.3f}s in itr: {i}, remove_nodes_from_adj.")

        # 4. Update partition:
        start = time.time()
        for e in range(row.shape[0]):
            r = row[e].item()
            c = col[e].item()
            uf.union(r, c)
        logger.debug(f"Spent: {time.time() - start:.3f}s in itr: {i}, union_find.")

    labels = []
    for n in range(num_nodes):
        labels.append(uf[n])
    return labels

