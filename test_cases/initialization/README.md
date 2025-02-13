# Initialization tests

These tests verify that nominal parameter values are read correctly and that `initializationPriorType` is correctly implemented. To be noticed, these tests covers that, following the SciML extension, users can set values and/or priors for specific layers (e.g., via `netId.layerId` in the parameters table) or parameter arrays (e.g., `netId.layerId.bias`).

For the test cases, either the nominal values are checked, or—when testing start guesses that are randomly sampled—the mean and variance of many samples are tested. What is tested, as well as the number of samples to take in stochastic tests, can be found in the `solutions.yaml` file for each test.
