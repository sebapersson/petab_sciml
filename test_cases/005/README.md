# Test Case 005

Test case for when a feed-forward neural network appears in one of the observable formulas.

## Model Structure

The SBML model for this problem is given as:

$$\frac{\mathrm{d} \text{prey}}{\mathrm{d} t} = \alpha \cdot \text{prey} - \beta \cdot \text{prey} \cdot \text{predator} $$

$$\frac{\mathrm{d} \text{predator}}{\mathrm{d} t} = \gamma \cdot \text{prey} \cdot \text{predator} - \delta \cdot \text{predator}$$

And the observable formula for `predator` is given by `NN[1]` (the output from a neural network).

## Data-Driven Model Structure

`N` is a feed-forward neural network with the following specification:

| LayerID | Layer                                                  | Activation Function |
|---------|--------------------------------------------------------|---------------------|
| layer1  | Linear(in_features = 2, out_features = 5, bias = true) | tanh                |
| layer2  | Linear(in_features = 5, out_features = 5, bias = true) | tanh                |
| layer3  | Linear(in_features = 5, out_features = 1, bias = true) | identity            |

The inputs to the network are given by the `prey` and `predator` species.

## Thoughts About Specification for This Case

When the data-driven model does not appear in the ODE, but rather in the observable function, we should allow direct encoding of the neural network in the observable formula. For example, the following should be allowed:

| observableId | observableFormula |
|--------------|-------------------|
| predator     | `net1[1]`         |

Then in the mapping table we can put:

| netId | ioId    | ioValue  |
|-------|---------|----------|
| net1  | input1  | prey     |
| net1  | input2  | predator |
| net1  | output1 | net1     |

By having the output here be `net1`, an importer can infer that the output should match a location where the network is encoded by itself, which should only happen in noise and observable formulas. Therefore, it will recognize that the network should not be inserted into the ODE model. In general, this should be the case when the output and/or the input does not depend on the ODE model RHS in any way.
