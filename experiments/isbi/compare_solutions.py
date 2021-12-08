import argparse
import h5py
import napari
import torch
from torch_em.util.util import get_trainer


def _predict(raw, ckpt):
    with torch.no_grad():
        model = get_trainer(ckpt).model
        model.eval()
        input_ = torch.from_numpy(raw[None, None]).to(torch.device("cuda"))
        pred = model(input_)[0].cpu().numpy()
    return pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input")
    args = parser.parse_args()

    # load the input data (last slice which is used for validation)
    with h5py.File(args.input, "r") as f:
        raw = f["raw"][-1]
    raw = raw.astype("float32") / 255.0

    # predict with the baseline model
    baseline = _predict(raw, "./checkpoints/baseline-model")

    # predict with the rama model
    rama = _predict(raw, "./checkpoints/rama-model")

    v = napari.Viewer()
    v.add_image(raw)
    v.add_image(baseline)
    v.add_image(rama)
    napari.run()


if __name__ == "__main__":
    main()
