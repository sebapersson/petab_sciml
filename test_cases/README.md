# PEtab SciML Extension Test Cases

This directory contains the test cases for the PEtab SciML extension. The tests are divided into three parts:

* Neural network import
* Hybrid models that combine mechanistic and data driven components
* Initialization (start guesses for parameter estimation)

In the neural network tests, a relatively large set of architectures are tested. Meanwhile, the hybrid tests include fewer network architectures. This is because if the hybrid interface works with a given library (e.g., Equinox, Lux.jl), and the implementation properly separates the neural network and dynamic components, the combination should function correctly given that network models can be properly imported.

Note that for neural networks, inputs, parameters (including gradients), and potential outputs are provided in the HDF5 file format, where arrays (e.g. parameter arrays) are stored in row-major format following PyTorch indexing. Therefore, in addition to accounting for the indexing, importers in column-major languages (e.g., Julia, R) need to account for the memory ordering.

## Neural-network tests

The neural networks test different layers and activation functions.

For each case, it is tested that the neural network computes the correct output given an input and a specified parameter weight table, for three different random input combinations. When building the tests, consistency is tested between Lux.jl and PyTorch. The following layers and activation functions are currently supported in the standard:

* **Layers**: `Conv1-3d`, `ConvTranspose1-3d`, `AvgPool1-3d`, `MaxPool1-3d`, `LPPool1-3d`, `AdaptiveMaxPool1-3d`, `AdaptiveMeanPool1-3d`, `BatchNorm1-3d`, `InstanceNorm1-3d`, `LayerNorm`, `Dropout1-3d`, `Dropout`, `AlphaDropout`, `Linear`, `Bilinear` and `Flatten`.
* **Activation**: `relu`, `relu6`, `hardtanh`, `hardswish`, `selu`, `leaky_relu`, `gelu`, `logsigmoid`, `tanhshrink`, `softsign`, `softplus`, `tanh`, `sigmoid`, `hardsigmoid`, `mish`, `elu`, `celu`, `softmax` and `log_softmax`.

## Hybrid models test

For each case, the following things are tested:

* That the model likelihood evaluates correctly.
* That the simulated values are correct from solving the model forward in time.
* Gradient correctness. Especially with SciML models, computing the gradient can be challenging as computing the Jacobian of the ODE model can be tricky, or to get the gradient of a neural network that sets model parameters.

## Initialization tests

These tests ensure that nominal parameter values are read correctly and that `initializationPriorType` is properly implemented. For the test cases, either the nominal values are directly verified, or, when testing start guesses that are randomly sampled, the mean and variance of multiple samples are evaluated.
