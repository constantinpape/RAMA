import os
from functools import partial

import numpy as np
import torch
import torch_em
from torch_em.model import UNet2d
from torch_em.data.datasets import get_isbi_loader
from torch_em.util import parser_helper

from multicut_loss import MulticutAffinityLoss
from utils import MulticutRandMetric

OFFSETS = [
    [-1, 0], [0, -1],
    [-3, 0], [0, -3],
    [-9, 0], [0, -9],
    [-27, 0], [0, -27]
]


def get_model(pretrained):
    n_out = len(OFFSETS)

    # we don't use any activation, as RAMA expects costs in range ]-inf,inf[ instead of [0,1]
    # so we could either use no activation or a Sigmoid followed by rescaling into ]-inf,inf[
    # the latter might be beneficial because it ensures costs in a sensible range in the beginning of training
    # final_activation = "Sigmoid"
    final_activation = None

    model = UNet2d(
        in_channels=1,
        out_channels=n_out,
        final_activation=final_activation,
    )
    if pretrained:
        assert os.path.exists(args.pretrained)
        with torch.no_grad():
            state = torch.load(args.pretrained, map_location="cpu")["model_state"]
            model.load_state_dict(state)
    return model


def train_rama(input_path, n_iterations, pretrained, device):
    model = get_model(pretrained)

    # shape of input patches (blocks) used for training
    patch_shape = [1, 512, 512]
    # patch_shape = [1, 64, 64]
    batch_size = 1

    normalization = partial(torch_em.transform.raw.normalize, minval=0, maxval=255)

    roi_train = np.s_[:28, :, :]
    train_loader = get_isbi_loader(
        input_path,
        download=True,
        patch_shape=patch_shape,
        rois=roi_train,
        batch_size=batch_size,
        raw_transform=normalization,
        num_workers=8*batch_size,
        n_samples=50,
        shuffle=True,
    )
    roi_val = np.s_[28:, :, :]
    val_loader = get_isbi_loader(
        input_path,
        download=False,
        patch_shape=patch_shape,
        rois=roi_val,
        batch_size=batch_size,
        raw_transform=normalization,
        num_workers=8*batch_size,
        n_samples=2,
        shuffle=True,
    )

    # with this scale we get (roughly) the 1/20 scale
    min_scale = 0.05
    max_scale = 0.5
    loss = MulticutAffinityLoss(patch_shape, OFFSETS, min_scale, max_scale, num_grad_samples=1)
    metric = MulticutRandMetric(OFFSETS)

    name = f"rama-model-v{args.version}"
    trainer = torch_em.default_segmentation_trainer(
        name=name,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss=loss,
        metric=metric,
        learning_rate=1e-4,
        mixed_precision=False,
        log_image_interval=50,
        device=device,

    )

    trainer.fit(n_iterations)


if __name__ == '__main__':
    parser = parser_helper()
    parser.add_argument("-v", "--version", required=True, type=int)
    parser.add_argument("-p", "--pretrained", default=None)
    args = parser.parse_args()
    train_rama(args.input, args.n_iterations, args.pretrained, args.device)
