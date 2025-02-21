# Published example: Boehm with universal observable mapping

- FNN replaces the observable mapping for `rSTAT5A_rel`
- published in UDEs for systems bio as *Boehm Scenario 4*

## Model structure

Boehm model for STAT5 dimerisation with a FNN for one of the observable mappings.

$$
\text{rSTAT5A\_rel}(t) = N
$$

## Data-Driven Model Structure

The input to the FNN are all eight dynamic species, the output is the `net1_output1` placeholder parameter. Specification:

| LayerID | Layer                                                  | Activation Function |
|---------|--------------------------------------------------------|---------------------|
| layer1  | Linear(in_features = 8, out_features = 5, bias = true) | tanh                |
| layer2  | Linear(in_features = 5, out_features = 5, bias = true) | tanh                |
| layer3  | Linear(in_features = 5, out_features = 1, bias = true) | tanh                |
