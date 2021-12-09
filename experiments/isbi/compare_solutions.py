import argparse
import h5py
import napari
import torch
from torch_em.util.util import get_trainer


def _predict(raw, ckpt):
    device = torch.device("cpu")
    with torch.no_grad():
        model = get_trainer(ckpt, device=device).model
        model.eval()
        input_ = torch.from_numpy(raw[None, None]).to(device)
        pred = model(input_)[0].cpu().numpy()
    return pred


# TODO segment with rama
def _segment(pred):
    pass


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
        baseline_seg = _segment(baseline)

    # predict with the rama model
    print("Run prediction for rama model version", args.version)
    rama = _predict(raw, f"./checkpoints/rama-model-v{args.version}")
    if bool(args.segment):
        rama_seg = _segment(rama)

    v = napari.Viewer()
    v.add_image(raw)
    v.add_image(baseline)
    v.add_labels(baseline_seg)
    v.add_image(rama)
    v.add_labels(rama_seg)
    napari.run()


if __name__ == "__main__":
    main()
