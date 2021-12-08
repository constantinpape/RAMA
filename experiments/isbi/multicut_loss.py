import nifty
import numpy as np
import rama_py
import rand_index_py
import torch
import torch.nn as nn

###
# this file should be moved to torch-em once everything fully works so it can be used in other experiments
# partially copied from RAMA/src/multicut_layer.py, which is not importable with the rama_py installation
###


def get_edge_labels(node_labels, uv_ids):
    uv_ids_long = uv_ids.long()
    edge_labels = node_labels[uv_ids_long[:, 0]] != node_labels[uv_ids_long[:, 1]]
    return edge_labels.to(torch.float32)


def compute_rand_index(node_labels_gt, incorrect_edge_labels_pred, incorrect_edge_indices):
    incorrect_edge_labels_gt = get_edge_labels(node_labels_gt, incorrect_edge_indices)
    norm = (2.0 * node_labels_gt.shape[0] * node_labels_gt.shape[0])
    return torch.abs(incorrect_edge_labels_pred - incorrect_edge_labels_gt).sum() / norm


def solve_multicut(uv_ids, edge_costs, solver_opts, contains_duplicate_edges=False):
    assert(uv_ids.shape[0] == edge_costs.shape[0])
    assert(uv_ids.shape[1] == 2)
    uv_ids_cpu = uv_ids.cpu().numpy()
    edge_costs_cpu = edge_costs.cpu().numpy()
    node_labels_cpu = rama_py.rama_cuda(
        uv_ids_cpu[:, 0], uv_ids_cpu[:, 1], edge_costs_cpu, solver_opts, contains_duplicate_edges
    )[0]
    node_labels = torch.IntTensor(node_labels_cpu).to(edge_costs.device)
    return node_labels


class MultiCutSolverWithRandIndex(torch.autograd.Function):
    @staticmethod
    def forward(ctx, uv_ids, uv_costs, params, node_labels_gt=None):
        ctx.set_materialize_grads(False)
        node_labels = solve_multicut(uv_ids, uv_costs, params["solver_opts"])
        uv_edge_labels = get_edge_labels(node_labels, uv_ids)
        if node_labels_gt is None:
            assert(~uv_costs.requires_grad)
        else:
            assert(node_labels.shape == node_labels_gt.shape)
            node_labels_cpu = node_labels.cpu().numpy()
            node_labels_gt_cpu = node_labels_gt.cpu().numpy()
            incorrect_edge_i, incorrect_edge_j = rand_index_py.compute_incorrect_edge_indices(
                node_labels_cpu, node_labels_gt_cpu
            )
            # Compute incorrect edge labels and indices.
            incorrect_edge_indices = torch.stack((
                torch.IntTensor(incorrect_edge_i), torch.IntTensor(incorrect_edge_j)
            ), 1).to(uv_costs.device)
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
        assert(ctx.params["num_grad_samples"] > 0)
        # This can create duplicate edges so multicut solver must merge duplicates first.
        merged_edge_indices = torch.cat((uv_ids, incorrect_edge_indices), 0)
        for s in range(ctx.params["num_grad_samples"]):
            loss_scaling = np.random.uniform(low=ctx.params["min_pert"], high=ctx.params["max_pert"])
            incorrect_edge_costs = loss_scaling * grad_incorrect_edge_labels
            merged_pert_edge_costs = torch.cat((uv_costs, incorrect_edge_costs), 0)
            node_labels_pert = solve_multicut(
                merged_edge_indices, merged_pert_edge_costs, ctx.params["solver_opts"], True
            )
            uv_edge_labels_pert = get_edge_labels(node_labels_pert, uv_ids)
            current_grad = (uv_edge_labels_pert - uv_edge_labels) / loss_scaling
            if s == 0:
                grad_avg = current_grad
            else:
                grad_avg += current_grad

            # the incorrect edge costs should be ~ 1/20 of costs (controlled by loss_scaling)
            # uncomment to monitor these values
            # print("costs:", uv_costs.min(), uv_costs.max())
            # print("incorrect-costs:", incorrect_edge_costs.min(), incorrect_edge_costs.max())
        grad_avg /= ctx.params["num_grad_samples"]
        assert(grad_avg.shape == uv_costs.shape)
        return None, grad_avg, None, None, None


class MulticutModuleWithRandIndex(torch.nn.Module):
    """
    Torch module for Multicut Instances. Only implemented for one multicut instance for now (batch-size 1)
    """
    def __init__(self, loss_min_scaling, loss_max_scaling, num_grad_samples):
        """
        loss_min_scaling: Minimum value of pertubation.
            Actual value is sampled in [loss_min_scaling, loss_max_scaling].
        loss_max_scaling: Maximum value of pertubation.
            Actual value is sampled in [loss_min_scaling, loss_max_scaling].
        num_grad_samples: Number of times to average the gradients by sampling loss scalar.
        """
        super().__init__()
        solver_opts = rama_py.multicut_solver_options("PD")
        solver_opts.verbose = False
        self.params = {"solver_opts": solver_opts,
                       "min_pert": loss_min_scaling,
                       "max_pert": loss_max_scaling,
                       "num_grad_samples": num_grad_samples}
        self.solver = MultiCutSolverWithRandIndex()

    def forward(self, uv_ids, uv_costs, node_labels_gt=None):
        return self.solver.apply(uv_ids, uv_costs, self.params, node_labels_gt)


class MulticutAffinityLoss(nn.Module):
    """Compute segmentation loss with rand index metric through RAMA.
    """
    def __init__(self, patch_shape, offsets, loss_min_scaling=1.0, loss_max_scaling=10.0, num_grad_samples=5):
        super().__init__()
        self.offsets = offsets
        self.ndim = len(offsets[0])
        if len(patch_shape) > self.ndim and patch_shape[0] == 1:
            self.patch_shape = tuple(patch_shape[1:])
        else:
            self.patch_shape = tuple(patch_shape)
        assert len(self.patch_shape) == self.ndim
        self._compute_graph_structure()
        self.multicut_layer = MulticutModuleWithRandIndex(loss_min_scaling, loss_max_scaling, num_grad_samples)

        # all torch_em classes should store init kwargs to easily recreate the init call
        self.init_kwargs = {"patch_shape": patch_shape, "offsets": offsets,
                            "loss_min_scaling": loss_min_scaling, "loss_max_scaling": loss_max_scaling,
                            "num_grad_samples": num_grad_samples}

    # NOTE we could improve performance by sending self.uv_ids, and self.edge_mask to the device already
    # (but for this would need to pass the device to __init__)
    # compute the graph structure for the given patch_shape from the offsets
    def _compute_graph_structure(self):
        # compute the grid graph
        g = nifty.graph.gridGraph(self.patch_shape)
        self.n_nodes = g.numberOfNodes

        # compute the uv-ids for the given offset pattern
        dummy_shape = (len(self.offsets),) + self.patch_shape
        dummy = np.zeros(dummy_shape, dtype="float32")
        self.n_edges, uv_ids, _ = g.affinitiesToEdgeMapWithOffsets(dummy, offsets=self.offsets, strides=None)
        self.uv_ids = torch.from_numpy(uv_ids[:self.n_edges].astype("int64"))

        # compute mask and index vector to go from affinity predictions to 1d cost vector
        edge_ids = g.projectEdgeIdsToPixelsWithOffsets(self.offsets)
        assert edge_ids.max() + 1 == self.n_edges
        edge_mask = edge_ids.ravel() != -1
        edge_ids = edge_ids.ravel()[edge_mask]
        assert edge_ids[0] == 0
        assert ((edge_ids[1:] - edge_ids[:-1]) == 1).all(), "Edge ids are not consecutive"
        self.edge_mask = torch.from_numpy(edge_mask)

    def forward(self, prediction, gt_segmentation):
        assert tuple(gt_segmentation.shape[:2]) == (1, 1),\
            f"Expected single batch and channel for gt_segmentation, got {gt_segmentation.shape}"
        assert tuple(gt_segmentation.shape[2:]) == self.patch_shape, f"{gt_segmentation.shape}, {self.patch_shape}"
        exp_pred_shape = (1, len(self.offsets),) + self.patch_shape
        assert tuple(prediction.shape) == exp_pred_shape, f"{prediction.shape}, {exp_pred_shape}"

        # map the segmentation to node_labels (can simply flatten)
        node_labels = gt_segmentation.flatten()
        assert len(node_labels) == self.n_nodes

        # map the prediction of shape (n_offsets,) + patch_shape to costs of shape (n_edges,)
        costs = prediction.flatten()[self.edge_mask.to(prediction.device)]
        assert len(costs) == self.n_edges

        node_labels_pred, incorrect_edge_labels_pred, incorrect_edge_indices = self.multicut_layer(
            self.uv_ids.to(prediction.device),
            costs,
            node_labels
        )
        loss = compute_rand_index(node_labels, incorrect_edge_labels_pred, incorrect_edge_indices)
        return loss
