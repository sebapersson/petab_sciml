# Test Suite

The PEtab SciML format provides an extensive test suite to verify the correctness of tools that support the standard. The tests are divided into three parts, which are recomended to be run in this order:

1. Neural network import
2. Initialization (start guesses for parameter estimation)
3. Hybrid models that combine mechanistic and data-driven components

The neural network tests cover a relatively large set of architectures. The hybrid tests, by comparison, involve fewer network architectures, because if the hybrid interface works with a given library (e.g., Equinox or Lux.jl) and the implementation cleanly separates neural network and dynamic components, the combination should function correctly once network models can be imported properly.

## Neural-network import tests

The neural networks test different layers and activation functions. A complete list of tested layers and activation functions can be found [here](@ref layers_activation).

## Initialization tests

These tests ensure that nominal parameter values are read correctly and that `initializationPriorType` is properly implemented. For the test cases, either the nominal values are directly verified, or, when testing start guesses that are randomly sampled, the mean and variance of multiple samples are evaluated.

## Hybrid Models Test

For each case, the following aspects are tested:

- Correct evaluation of the model likelihood.
- Accuracy of simulated values when solving the model forward in time.
- Gradient correctness. This is particularly important for SciML models, where computing gradients can be challenging.

If a tool supports providing the neural network model in a format other than the YAML fromat, we recommend modifying the tests in this folder to use another neural network format to verify the correctness of the implementation.
