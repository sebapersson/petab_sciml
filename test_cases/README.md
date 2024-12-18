# PEtab SciML Extension Test Cases

This directory contains the test cases for the PEtab SciML extension. The tests are divided into two parts: those testing hybrid models and those testing pure neural network import. The rationale is that once the hybrid interface works with a suitable library (e.g., Equinox, Lux.jl...), then, as ideally implementation should modularize the neural network and dynamic parts, the combination should work.

It should be noted that for neural networks, inputs, parameters (including gradients), and potential outputs are provided in the HDF5 file format, where arrays are stored in row-major format following PyTorch indexing. Therefore, in addition to accounting for the indexing, importers in column-major languages (e.g., Julia, R) need to account for the memory ordering.

## Hybrid models test

For each case, the following things are tested:

* That the model likelihood evaluates correctly.
* That the simulated values are correct from solving the model forward in time.
* Gradient correctness. Especially with SciML models, computing the gradient can be challenging as computing the Jacobian of the ODE model can be tricky, or to get the gradient of a neural network that sets model parameters.

## Neural-network tests

The neural networks test different layers and activation functions. Neural network parameters (e.g., weights) are stored in tidy format, where the indexing corresponds to PyTorch. For example, `netId_layerId_weight_0_1_1` is the weight for `layerId` at index `(0, 1, 1)`, which, given how Python denotes tensors, would correspond to `(2, 2, 1)` in Julia. Similarly, for input, PyTorch conventions are respected. This means that for channel input (e.g., for an image), the channel is the first dimension (unless there is a batch dimension then it is first).

Important to consider for implementations is the `flatten` operation. By default, PyTorch flattens following a column-major ordering. For languages storing in row-major order, this should be respected to avoid non-reproducible results across flattening calls. This raises the question of the best way to handle this. The simplest and likely the most efficient approach, as it respects how arrays are stored in memory, is to transform the input data. For example, in PyTorch, image dimensions should be provided in the form `(C, W, H)`, while in Julia, it should be stored as `(H, W, C)`. If, during neural net import into Julia, the input data and the convolutional kernels (and similar layers) are transformed to respect the Julia format, the resulting output will be consistent following flattening. See, for example, test case `net_018`.

### What is tested

For each case, it is tested that the neural network computes the correct output given an input and a specified parameter weight table, for three different random input combinations. When building the tests, consistency is tested between Lux.jl and PyTorch. Basically, by testing that two frameworks can compute the same values, we verify whether it is possible to build a consistent importer.

The following layers and activation functions are currently supported in the standard:

* **Layers**: `Conv1-3d`, `ConvTranspose1-3d`, `AvgPool1-3d`, `MaxPool1-3d`, `LPPool1-3d`, `AdaptiveMaxPool1-3d`, `AdaptiveMeanPool1-3d`, `BatchNorm1-3d`, `InstanceNorm1-3d`, `LayerNorm`, `Dropout1-3d`, `Dropout`, `AlphaDropout`, `Linear`, `Bilinear` and `Flatten`.
* **Activation**: `relu`, `relu6`, `hardtanh`, `hardswish`, `selu`, `leaky_relu`, `gelu`, `logsigmoid`, `tanhshrink`, `softsign`, `softplus`, `tanh`, `sigmoid`, `hardsigmoid`, `mish`, `elu`, `celu`, `softmax` and `log_softmax`.
