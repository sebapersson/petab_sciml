# Test Case 015

Test case for when a feed-forward neural network sets one of the model parameters, with the input to the network being provided as a PyTorch file with PyTorch indexing. Since the network is not a part of the ODE model's right-hand side (RHS), it should only be evaluated once per likelihood computation for computational efficiency.

## Model Structure

The SBML model for this problem is given as:

$$\frac{\mathrm{d} \text{prey}}{\mathrm{d} t} = \alpha \cdot \text{prey} - \beta \cdot \text{prey} \cdot \text{predator} $$

$$\frac{\mathrm{d} \text{predator}}{\mathrm{d} t} = \gamma \cdot \text{predator} \cdot \text{prey} - \delta \cdot \text{predator}$$

## Data-Driven Model Structure

`N` is a feed-forward neural network with the following specification:

| LayerID | Layer                                                  | Activation Function |
|---------|--------------------------------------------------------|---------------------|
| layer1  | Conv3d(3, 1, (5, 5)) | identity                |
| layer2  | Flatten() | identity                |
| layer3  | Linear(in_features = 36, out_features = 1, bias = true) | relu            |

The inputs to the network are provided by the `input_data.tsv` file and the output replaces the `gamma` parameter.

## Additional notes

Providing input data as a TSV file can be impractical in certain applications, such as when data consists of large images. Therefore, the PEtab SciML standard does not impose a specific input data format. Rather, this and test-case 016 aims to ensure that an implementation can handle input provided via input files. Because, once it is possible to read one input into an array, adding support for other formats should be relatively straightforward. Alternatively, it is also relatively straightforward to allow users to provide their own data import functions, that takes the file path as input, and outputs a suitable array/tensor.
