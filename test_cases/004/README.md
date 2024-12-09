# Test Case 004

Test case for when a feed-forward network sets one of the model parameters, but the input to the network is given by two parameters that are defined in the conditions table. Since the network is not a part of the ODE model's RHS, it should only be evaluated once per simulation condition for computational efficiency.

## Model Structure

The SBML model for this problem is given as:

$$\frac{\mathrm{d} \text{prey}}{\mathrm{d} t} = \alpha \cdot \text{prey} - \beta \cdot \text{prey} \cdot \text{predator} $$

$$\frac{\mathrm{d} \text{predator}}{\mathrm{d} t} = \gamma \cdot \text{predator} - \delta \cdot \text{prey} \cdot \text{predator}$$

## Data-Driven Model Structure

`N` is a feed-forward neural network replacing the `gamma` parameter with the following specification:

| LayerID | Layer                                                  | Activation Function |
|---------|--------------------------------------------------------|---------------------|
| layer1  | Linear(in_features = 2, out_features = 5, bias = true) | tanh                |
| layer2  | Linear(in_features = 5, out_features = 5, bias = true) | tanh                |
| layer3  | Linear(in_features = 5, out_features = 1, bias = true) | identity            |

The inputs to the network are given by `input1` and `input2`, which take different values for different conditions.
