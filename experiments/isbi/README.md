# ISBI Experiments

Experiments on neurons segmentation data from the ISBI2012 segmentation challenge.
Currently, to run these experiments you need to install `torch_em` with its dependencies and RAMA on the `rand_index` branch.
Then run via
- `python train_isbi_rama.py -i /path/to/isbi.h5` for experiments with the multicut-rand-index loss
- `python train_isbi_baseline.py -i /path/to/isbi.h5` for the baseline experiments

The training data will be downloaded automatically to the given path.

## Evaluation

| Version  | Description | RandError  |
| -------- | ----------- | ---------- |
| baseline | -           | 0.0691     |
| v1       | e2e         | 0.3412     |
| v2       | e2e + pretr | 0.0671     |
