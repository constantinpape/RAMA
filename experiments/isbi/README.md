# ISBI Experiments

Experiments on neurons segmentation data from the ISBI2012 segmentation challenge.
Currently, to run these experiments you need to install `torch_em` with its dependencies and RAMA on the `rand_index` branch.
Then run via
- `python train_isbi_rama.py -i /path/to/isbi.h5` for experiments with the multicut-rand-index loss
- `python train_isbi_baseline.py -i /path/to/isbi.h5` for the baseline experiments

The training data will be downloaded automatically to the given path.

## Evaluation

| Version  | Description | Score  |
| -------- | ----------- | ------ |
| baseline | -           | 0.9171 |
| v1       | rama e2e    | 0.3412 |
| v2       | rama e2e + pretrained | 0.0671 |
