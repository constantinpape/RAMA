import argparse
import h5py
import napari
import torch
from torch_em.util.util import get_trainer
from utils import segment_rama


OFFSETS = [
    [-1, 0], [0, -1],
    [-3, 0], [0, -3],
    [-9, 0], [0, -9],
    [-27, 0], [0, -27]
]


def _predict(raw, ckpt):
    device = torch.device("cpu")
    with torch.no_grad():
        model = get_trainer(ckpt, device=device).model
        model.eval()
        input_ = torch.from_numpy(raw[None, None]).to(device)
        pred = model(input_)[0].cpu().numpy()
    return pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input")
    parser.add_argument("-v", "--version", default=1, type=int)
    parser.add_argument("-s", "--segment", default=0, type=int)
    args = parser.parse_args()

    # load the input data (last slice which is used for validation)
    with h5py.File(args.input, "r") as f:
        raw = f["raw"][-1]
    raw = raw.astype("float32") / 255.0

    # predict with the baseline model
    print("Run prediction for baseline model")
    baseline = _predict(raw, "./checkpoints/baseline-model")
    if bool(args.segment):
        print("Run segmentation for baseline model")
        baseline_seg = segment_rama(baseline, OFFSETS)

    # predict with the rama model
    print("Run prediction for rama model version", args.version)
    rama = _predict(raw, f"./checkpoints/rama-model-v{args.version}")
    if bool(args.segment):
        print("Run segmentation for rama model")
        rama_seg = segment_rama(rama, OFFSETS)

    v = napari.Viewer()
    v.add_image(raw)
    v.add_image(baseline)
    v.add_labels(baseline_seg)
    v.add_image(rama)
    v.add_labels(rama_seg)
    napari.run()


if __name__ == "__main__":
    main()
