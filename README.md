# RAMA: Rapid algorithm for multicut problem
Solves multicut (correlation clustering) problems orders of magnitude faster than CPU based solvers without compromising solution quality on NVIDIA GPU. It also gives lower bound guarantees.

![animation](./misc/contraction_animation.gif)

## Requirements
We use `CUDA 11.2` and `GCC 10`. Other combinations might also work but not tested. `CMake` is required for compilation.

## Running
Currently https://anonymous.4open.science/ does not allow adding submodules and downloading the repo. So current files can only be viewed, we will make full code available upon acceptance.

## Code highlights
- [src/rama_cuda.cu: ](src/rama_cuda.cu) Main solver entry point.
- [src/multicut_message_passing.cu:](src/multicut_message_passing.cu) Dual solver.
- [src/dCOO.cu](src/dCOO.cu#L53) Parallel edge contraction (primal solver).

