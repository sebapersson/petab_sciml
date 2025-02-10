# Test Case 014

Test case for neural-network parameter initialization. Here the neural network has `glorotUniform` `initializationPriorType` with `gain = 1.0`, except for `layer1` which has `glorotUniform` with `gain = 0.4`. Note, that during initialization of parameters more nested levels have precedence.

## Model Structure

The same model structure as in test case 001 for SciML import.
