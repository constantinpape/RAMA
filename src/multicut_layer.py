import numpy as np
import torch
import rama_py
import rand_index_py

def solve_multicut(uv_ids, edge_costs, solver_opts, contains_duplicate_edges = False):
    assert(uv_ids.shape[0] == edge_costs.shape[0])
    assert(uv_ids.shape[1] == 2)
    uv_ids_cpu = uv_ids.cpu().numpy()
    edge_costs_cpu = edge_costs.cpu().numpy()
    node_labels_cpu = rama_py.rama_cuda(uv_ids_cpu[:, 0], uv_ids_cpu[:, 1], edge_costs_cpu, solver_opts, contains_duplicate_edges)[0]
    node_labels = torch.IntTensor(node_labels_cpu).to(edge_costs.device)
    return node_labels

def get_edge_labels(node_labels, uv_ids):
    uv_ids_long = uv_ids.long()
    edge_labels = node_labels[uv_ids_long[:, 0]] != node_labels[uv_ids_long[:, 1]]
    return edge_labels.to(torch.float32)

class MultiCutSolverWithRandIndex(torch.autograd.Function):
    @staticmethod
    def forward(ctx, uv_ids, uv_costs, params, node_labels_gt = None):
        ctx.set_materialize_grads(False)
        node_labels = solve_multicut(uv_ids, uv_costs, params['solver_opts'])
        uv_edge_labels = get_edge_labels(node_labels, uv_ids)
        if node_labels_gt == None:
            assert(~uv_costs.requires_grad)
        else:
            assert(node_labels.shape == node_labels_gt.shape)
            node_labels_cpu = node_labels.cpu().numpy()
            node_labels_gt_cpu = node_labels_gt.cpu().numpy()
            incorrect_edge_i, incorrect_edge_j = rand_index_py.compute_incorrect_edge_indices(node_labels_cpu, node_labels_gt_cpu)
            # Compute incorrect edge labels and indices.
            incorrect_edge_indices = torch.stack((torch.IntTensor(incorrect_edge_i), torch.IntTensor(incorrect_edge_j)), 1).to(uv_costs.device)
            incorrect_edge_labels = get_edge_labels(node_labels, incorrect_edge_indices)

        ctx.params = params
        ctx.device = uv_costs.device
        ctx.save_for_backward(uv_costs, uv_ids, uv_edge_labels, incorrect_edge_labels, incorrect_edge_indices)

        ctx.mark_non_differentiable(node_labels)
        ctx.mark_non_differentiable(incorrect_edge_indices)
        return node_labels, incorrect_edge_labels, incorrect_edge_indices

    @staticmethod
    def backward(ctx, grad_node_labels, grad_incorrect_edge_labels, grad_edge_indices):
        """
        Backward pass computation.
        """
        uv_costs, uv_ids, uv_edge_labels, incorrect_edge_labels, incorrect_edge_indices = ctx.saved_tensors
        assert(grad_incorrect_edge_labels.shape == incorrect_edge_labels.shape)
        grad_avg = None
        assert(ctx.params['num_grad_samples'] > 0)
        merged_edge_indices = torch.cat((uv_ids, incorrect_edge_indices), 0) # This can create duplicate edges so multicut solver must merge duplicates first.
        for s in range(ctx.params['num_grad_samples']):
            loss_scaling = np.random.uniform(low = ctx.params['min_pert'], high = ctx.params['max_pert'])
            incorrect_edge_costs = loss_scaling * grad_incorrect_edge_labels
            merged_pert_edge_costs = torch.cat((uv_costs, incorrect_edge_costs), 0)
            node_labels_pert = solve_multicut(merged_edge_indices, merged_pert_edge_costs, ctx.params['solver_opts'], True)
            uv_edge_labels_pert = get_edge_labels(node_labels_pert, uv_ids)
            current_grad = (uv_edge_labels_pert - uv_edge_labels) / loss_scaling
            if s == 0:
                grad_avg = current_grad
            else:
                grad_avg += current_grad
        grad_avg /= ctx.params['num_grad_samples']
        assert(grad_avg.shape == uv_costs.shape)
        return None, grad_avg, None, None, None

class MulticutModuleWithRandIndex(torch.nn.Module):
    """
    Torch module for Multicut Instances. Only implemented for one multicut instance for now (batch-size 1)
    """
    def __init__(self, loss_min_scaling, loss_max_scaling, num_grad_samples):
        """
        loss_min_scaling: Minimum value of pertubation. Actual value would be sampled in [loss_min_scaling, loss_max_scaling].
        loss_max_scaling: Maximum value of pertubation. Actual value would be sampled in [loss_min_scaling, loss_max_scaling].
        num_grad_samples: Number of times to average the gradients by sampling loss scalar in range [loss_min_scaling, loss_max_scaling].
        """
        super().__init__()
        solver_opts = rama_py.multicut_solver_options("PD")
        solver_opts.verbose = False
        self.params = {'solver_opts': solver_opts, 
                        'min_pert': loss_min_scaling, 
                        'max_pert': loss_max_scaling,
                        'num_grad_samples': num_grad_samples}
        self.solver = MultiCutSolverWithRandIndex()

    def forward(self, uv_ids, uv_costs, node_labels_gt = None):
        return self.solver.apply(uv_ids, uv_costs, self.params, node_labels_gt)

def compute_rand_index(node_labels_gt, incorrect_edge_labels_pred, incorrect_edge_indices):
    incorrect_edge_labels_gt = get_edge_labels(node_labels_gt, incorrect_edge_indices)
    return torch.abs(incorrect_edge_labels_pred - incorrect_edge_labels_gt).sum() / (2.0 * node_labels_gt.shape[0] * node_labels_gt.shape[0])

if __name__ == "__main__":
    multicut_layer = MulticutModuleWithRandIndex(1.0, 10.0, 5)
    i = torch.IntTensor([0, 1, 2, 3, 4]).cuda()
    j = torch.IntTensor([1, 2, 3, 4, 0]).cuda()
    uv_ids = torch.stack((i, j), 1)
    costs = torch.FloatTensor([1, 2.2, 0.4, 10.1, 0.3]).cuda()
    costs.requires_grad = True 
    node_labels_gt = torch.IntTensor([0, 0, 0, 1, 1]).cuda()
    node_labels_pred, incorrect_edge_labels_pred, incorrect_edge_indices = multicut_layer(uv_ids, costs, node_labels_gt)
    loss = compute_rand_index(node_labels_gt, incorrect_edge_labels_pred, incorrect_edge_indices)
    loss.backward()
    print(f"Initial loss: {loss.item()}")
    costs_new = costs.detach().clone() - 20 * costs.grad # Take a step towards decreasing loss.
    _, incorrect_edge_labels_pred, incorrect_edge_indices = multicut_layer(uv_ids, costs_new, node_labels_gt)
    loss = compute_rand_index(node_labels_gt, incorrect_edge_labels_pred, incorrect_edge_indices)
    print(f"Loss after one step: {loss.item()}")


