# Test Case 012

Test case with two feed-forward neural networks where one is inside the ODE and the other sets a model parameter.

## Model Structure

Lotka-Volterra model with two neural networks:

$$\frac{\mathrm{d} \text{prey}}{\mathrm{d} t} = \alpha \cdot \text{prey} - net2 \cdot \text{prey} \cdot \text{predator} $$

$$\frac{\mathrm{d} \text{predator}}{\mathrm{d} t} = net1 - \delta \cdot \text{predator}$$

The SBML model for this problem is given as:

$$\frac{\mathrm{d} \text{prey}}{\mathrm{d} t} = \alpha \cdot \text{prey} - \beta \cdot \text{prey} \cdot \text{predator} $$

$$\frac{\mathrm{d} \text{predator}}{\mathrm{d} t} = \gamma - \delta \cdot \text{predator}$$

Hence, the neural network replaces $\gamma$ and $\beta$ during the import.

## Data-Driven Model Structure

`net1` is a feed-forward neural network with the following specification:

| LayerID | Layer                                                  | Activation Function |
|---------|--------------------------------------------------------|---------------------|
| layer1  | Linear(in_features = 2, out_features = 5, bias = true) | tanh                |
| layer2  | Linear(in_features = 5, out_features = 5, bias = true) | tanh                |
| layer3  | Linear(in_features = 5, out_features = 1, bias = true) | identity            |

The inputs to the network are provided by the `prey` and `predator` parameters from the parameters table.

`net2` is a feed-forward neural network with the following specification:

| LayerID | Layer                                                  | Activation Function |
|---------|--------------------------------------------------------|---------------------|
| layer1  | Linear(in_features = 2, out_features = 5, bias = true) | relu                |
| layer2  | Linear(in_features = 5, out_features = 10, bias = true) | relu                |
| layer3  | Linear(in_features = 10, out_features = 1, bias = true) | identity            |

The inputs to the network are provided by the `input1` and `input2` parameters from the parameters table.
