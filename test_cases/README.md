# PEtab SciML Extension Test Cases

This directory contains the test cases for the PEtab SciML extension. The tests are divided into two parts: those testing hybrid models and those testing pure neural network import. The rationale is that once the hybrid interface works with a suitable library (e.g., Equinox, Lux.jl...), then, as ideally implementation should modularize the neural network and dynamic parts, the combination should work.

## Hybrid models test

For each case, the following things are tested:

* That the data-driven (typically neural network) model evaluates correctly. This means that any importer should support the property to separately compute the data-driven model for a given input.
* That the model likelihood evaluates correctly.
* That the simulated values are correct from solving the model forward in time.
* Gradient correctness. Especially with SciML models, computing the gradient can be challenging as deriving the Jacobian of the ODE model can be tricky. For example, without an automatic differentiation framework it can be hard to differentiate data-driven models that are not a part of the ODE model.

Below is a very brief summary of each case; more details can be found in the corresponding directory.

## Case 001

A feed-forward neural network is a part of the ODE model.

## Case 002

A feed-forward neural network sets one of the derivatives of the ODE model.

## Case 003

A feed-forward neural network sets the values for a subset of ODE model kinetic parameters. As the neural network is not part of the ODE right-hand side, it only needs to be evaluated once per likelihood computation.

## Case 004

A feed-forward neural network sets the values for a subset of ODE model kinetic parameters, where the input to the neural network depends on the simulation condition.

## Case 005

A feed-forward neural network appears only in one of the observable formulas.

## Neural-network tests

The neural networks test different layers and activation functions. Neural network parameters (e.g., weights) are stored in tidy format, where the indexing corresponds to PyTorch. For example, `netId_layerId_weight_0_1_1` is the weight for `layerId` at index `(0, 1, 1)`, which, given how Python denotes tensors, would correspond to `(2, 2, 1)` in Julia. Similarly, for input, PyTorch conventions are respected. This means that for channel input (e.g., for an image), the channel is the first dimension (unless there is a batch dimension then it is first).

Important to consider for implementations is the `flatten` operation. By default, PyTorch flattens following a column-major ordering. For languages storing in row-major order, this should be respected. Otherwise, results are non-reproducible across flattening calls.

For each case, it is tested that the neural network computes the correct output given an input and a specified parameter weight table, for three different random input combinations. When building the tests, consistency is tested between Lux.jl and PyTorch. Basically, by testing that two frameworks can compute the same values, we verify whether it is possible to build a consistent importer.

The following layers and activation functions are currently supported in the standard:

* **Layers**: `Conv1-3d`, `ConvTranspose1-3d`, `AvgPool1-3d`, `MaxPool1-3d`, `LPPool1-3d`, `AdaptiveMaxPool1-3d`, `AdaptiveMeanPool1-3d`, `Dropout1-3d`, `Dropout`, `AlphaDropout`, `Linear`, `Bilinear` and `Flatten`.
* **Activation**: `relu`, `relu6`, `hardtanh`, `hardswish`, `selu`, `leaky_relu`, `gelu`, `logsigmoid`, `tanhshrink`, `softsign`, `softplus`, `tanh`, `sigmoid`, `hardsigmoid`, `mish`, `elu`, `celu`, `softmax` and `log_softmax`.
