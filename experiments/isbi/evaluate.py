import argparse
import h5py
import torch
from elf.evaluation import rand_index
from torch_em.util.util import get_trainer
from utils import segment_rama


OFFSETS = [
    [-1, 0], [0, -1],
    [-3, 0], [0, -3],
    [-9, 0], [0, -9],
    [-27, 0], [0, -27]
]


def _segment(raw, ckpt):
    device = torch.device("cpu")
    with torch.no_grad():
        model = get_trainer(ckpt, device=device).model
        model.eval()
        input_ = torch.from_numpy(raw[None, None]).to(device)
        pred = model(input_)[0].cpu().numpy()
    seg = segment_rama(pred, OFFSETS)
    return seg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input")
    parser.add_argument("-c", "--checkpoint")
    args = parser.parse_args()
    ckpt = args.checkpoint

    # load the input data (last slice which is used for validation)
    with h5py.File(args.input, "r") as f:
        raw = f["raw"][-1]
        gt = f["labels/gt_segmentation"][-1]
    raw = raw.astype("float32") / 255.0
    seg = _segment(raw, ckpt)
    score = rand_index(seg, gt)[0]
    print("Eval score for", ckpt, score)


if __name__ == "__main__":
    main()
