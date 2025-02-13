# Format Specification

A PEtab SciML problem extends the PEtab standard version 2 to accommodate hybrid models SciML problems that combine data-driven (neural net) and mechanistic components. The extension introduces one new PEtab file type:

1. **Neural Net File(s)**: YAML file(s) describing neural net model(s).

It further extends the following standard PEtab files:

1. **Mapping Table**: Used to describe how neural network inputs and outputs map to PEtab quantities.
2. **Parameters Table**: Used to describe nominal values and potential priors for initializing network parameters.
3. **Condition Table**: Used to assign neural network outputs and inputs.
4. **Problem YAML File**: Includes a new SciML field.

All other PEtab files remain unchanged. This page explains the format and options for each file that is added or modified by the PEtab SciML extension.

## High Level Overview

The main goal of the PEtab SciML extension is to enable hybrid models that combine data-driven and mechanistic components. There are three types of hybrid model considered, each specified differently:

1. **Data-driven models in the ODE model’s right-hand side (RHS):** In this scenario, the SBML file is modified during import by either replacing a derivative or assigning a parameter to a neural network output. In both cases, the neural network input and output variables (as defined in the mapping table) must be assigned in the condition table using the `setNetRate` and/or `setNetAssignment` operator types.

2. **Data-driven models in the observable function:** In this scenario, the neural network output variable (as defined in the mapping table) is directly embedded in the observable formula. Meanwhile, the input variables (also defined in the mapping table) are assigned in the condition table using the `setNetAssignment` operator type.

3. **Data-driven models before the ODE model:** In this scenario, the data-driven model sets constant parameters or initial values in the ODE model prior to simulation. The input variable (as defined in the mapping table) can be assigned in the parameter or condition table as a standard constant PEtab variable, and the output variables (as defined in the mapping table) are assigned via the condition table.

## Neural Network Model Format

**TODO:** Dilan, I will need some help from you here, link the scheme?

Neural network models are provided as separate YAML files, and each tool supporting the extension is responsible for importing this file into a suitable format. A neural network YAML file has two main sections:

- **layers**: Defines the neural network layers, each with a unique ID. The layer names and argument syntax follow PyTorch conventions.
- **forward**: Describes the forward pass, specifying the order of layer calls and any applied activation functions.

Although network YAML files can be manually written, the recommended approach is to define a PyTorch `nn.Module` whose constructor sets up the layers and whose `forward` method specifies how they are invoked. For example, a simple feed-forward network can be defined as:

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(in_features=2, out_features=5)
        self.layer2 = nn.Linear(in_features=5, out_features=5)
        self.layer3 = nn.Linear(in_features=5, out_features=1)

    def forward(self, x):
        x = self.layer1(x)
        x = F.tanh(x)
        x = self.layer2(x)
        x = F.tanh(x)
        x = self.layer3(x)
        return x
```

The corresponding YAML file can then be generated using the `petab_sciml` library:

```python
# TODO: Add
```

Any PyTorch-supported keyword can be supplied for each layer in the YAML file, allowing for a broad range of architectures. For example, a more complex convolutional model might be structured as:

```python
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.max_pool1 = nn.MaxPool2d((2, 2))
        self.fc1 = nn.Linear(64, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.flatten1 = nn.Flatten()

    def forward(self, input):
        c1 = self.conv1(input)
        s2 = self.max_pool1(c1)
        c3 = self.conv2(s2)
        s4 = self.max_pool1(c3)
        s4 = self.flatten1(s4)
        f5 = self.fc1(s4)
        f6 = self.fc2(f5)
        output = self.fc3(f6)
        return output
```

A complete list of supported and tested layers and activation functions can be found [here](@ref layers_activation).

### Neural Network Parameters

All parameters for a neural network model are stored in an HDF5 file, with the file name specified in the parameter table. In this file, each layer’s parameters are grouped under `f.layerId`, where `layerId` is the layer’s unique identifier. For example, the weights of a linear layer are stored at `f.layerId.weight`.

Since parameters are stored in an HDF5 format, they are stored as arrays. The indexing follows PyTorch conventions, meaning parameters are stored in row-major order. Typically, users do not need to manage these details manually, as PEtab SciML tools should handle them automatically.

### Neural Network Input

When network input is provided as an array via an HDF5 file (see the mapping table below), it should follow the PyTorch convention. For example, if the first layer is `Conv2d`, the input should be in `(C, W, H)` format, with data stored in row-major order. In general, the input should be structured to be directly compatible with PyTorch.

!!! tip "For developers: Respect memory order"
    Tools supporting the SciML extension should, for computational efficiency, reorder input data and potential layer parameter arrays to match the memory ordering of the target language. For example, PEtab.jl converts input data to column-major order, as used in Julia.

## Mapping Table

The mapping table describes whcih PEtab problem variables a neural network’s inputs and outputs map. Each neural network input and output must be mapped in the mapping table.

| **petabEntityId** | **modelEntityId** |
|-------------------|-------------------|
| e.g.              |                   |
| k1                | netId.input1      |

### Detailed Field Descriptions

- **petabEntityId [STRING]**: A valid PEtab identifier not defined elsewhere in the PEtab problem. It can be referenced in the condition, measurement, parameter, and observable tables or be a file, but not in the model itself. For neural network outputs, the PEtab identifier must be assigned in the condition table, whereas for inputs, this is not required (see examples below).

- **modelEntityId [STRING]**: Describes the neural network entity corresponding to the `petabEntityId`. Must follow the format `netId.input{n}` or `netId.output{n}`, where `n` is the specific input or output index.

### Network with Scalar Inputs

For networks with scalar inputs, the PEtab entity should be a PEtab variable. For example, assume that the network `net1` has two inputs, then a valid mapping table would be:

| **petabEntityId** | **modelEntityId** |
|-------------------|-------------------|
| net1_input1       | net1.input1       |
| net1_input2       | net1.input2       |

Scalar input variables can then be:

- **Parameters in the parameters table**: These may be either estimated or constant.
- **Parameters assigned in the condition table**: More details on this option can be found in the section on the condition table.

### Network with Array Inputs

Sometimes, such as with image data, a neural network requires array input. In these cases, the input can be specified as an HDF5 file path directly in the mapping table:

| **petabEntityId**   | **modelEntityId**  |
|-------------------|------------------|
| input_net1.hdf5    | net1.input1      |

As mentioned in [ADD], the HDF5 file should follow PyTorch indexing and be stored in row-major order.

When there are multiple simulation conditions that each require a different neural network array input, the mapping table should map to a PEtab variable (e.g., `net1_input`):

| **petabEntityId** | **modelEntityId**  |
|-----------------|------------------|
| net1_input      | net1.input1      |

This variable is then assigned to specific input files via the condition table using the `setValue` `operatorType`. For a full example of a valid PEtab problem with array inputs, see [ADD].

### Network Observable Formula Output

If the neural network output appears in the observable formula, the PEtab entity should be directly referenced in the observable formula. For example, given:

| **petabEntityId** | **modelEntityId**  |
|-----------------|-----------------|
| net1_output1    | net1.output1    |

A valid observable table would be:

| **observableId** | **observableFormula** |
|----------------|----------------------|
| obs1           | net1_output1        |

As usual, the `observableFormula` can be any valid PEtab equation, so `net1_output1 + 1` would also be valid.

### Network Scalar Output

If the output does not appear in the observable, the output variable should still be defined in the mapping table:

| **petabEntityId** | **modelEntityId**  |
|-----------------|-----------------|
| net1_output1    | net1.output1    |

The output parameter (`net1_output1`) is then assigned in the condition table (see below).

### Additional Details

Although a neural network can, in principle, accept both array and scalar inputs, this feature is not currently tested for among tools implementing the PEtab SciML extension due to it being hard to implement. However, tools are free to add this feature.

## Condition Table

In the PEtab SciML extension, the condition table is extended to specify how neural network outputs (and, if necessary, inputs) are assigned. Two new `operatorType` are introduced in the extension to support this functionality:

1. **setNetRate**: Assigns the rate of a species to a neural network output. Here, `targetValue` must be a neural network output and `targetId` must be a model specie.
2. **setNetAssignment**: Assigns the input or output of a neural network in the ODE right-hand side (RHS) or the input in the observable formula.
   - Input Case: `targetId` is a neural network input, and `targetValue` can be any valid PEtab math expression that references model variables.
   - Output Case: `targetId` is a non-estimated ODE model parameter, and `targetValue` is a neural network output. This is used to assign a neural network output in the ODE model RHS.

!!! warn "Model structure altering conditions"
    IWhen `setNetRate` or `setNetAssignment` are used during model import the generated model structure or observable formula is altered, basically a neural-network is inserted into the generated functions. Therefore, as unique model structures per condition are not supported in most PEtab tools, the same `setNetAssignment` or `setNetRate` assignment must be set per condition.

### Assigning Neural Network Output

To set a **constant model parameter value** before model simulations, the `setValue` operator type should be used. For example, if parameter `p` is determined by `net1_output1` (mapped to a neural network output in the mapping table), a valid condition table entry is:

| **conditionId** | **operatorType** | **targetId** | **targetValue** |
|-----------------|------------------|--------------|-----------------|
| cond1           | setValue         | p            | net1_output1    |

Note that this specification allows for condition-specific assignments. For example, `net1_output1` could target a different parameter in another condition or multiple model parameters in the same condition.

To set an **initial value**, the `setInitial` operator type should be used. For example, if the initial value of species `X` comes from `net1_output1`, a valid condition table entry is:

| **conditionId** | **operatorType** | **targetId** | **targetValue** |
|-----------------|------------------|--------------|-----------------|
| cond1           | setInitial       | X            | net1_output1    |

To set a **model derivative**, the `setNetRate` operator type should be used. For example, if the rate of species `X` is given by `net1_output1`, a valid condition table entry is:

| **conditionId** | **operatorType** | **targetId** | **targetValue** |
|-----------------|------------------|--------------|-----------------|
| cond1           | setNetRate       | X            | net1_output1    |

To alter the **ODE RHS**, the `setNetAssignment` operator type should be used. For example, if an ODE model parameter `p` should be given by `net1_output1`, a valid condition table entry is:

| **conditionId** | **operatorType**    | **targetId** | **targetValue** |
|-----------------|---------------------|--------------|-----------------|
| cond1           | setNetAssignment    | p            | net1_output1    |

### Assigning Neural Network Input

When a neural network sets a **constant model parameter value or initial value**, its input variable (as specified in the mapping table) is a standard PEtab variable. If that input variable is not defined in the parameter table, it should be assigned using the `setValue` operator type.

When a neural network sets **a model derivative or alters the ODE RHS**, the input typically depends on model entities. Therefore, the input variable should be assigned using the `setNetAssignment` operator. For example, if neural network input `net1_input1` is given by specie `X`, a valid condition table is:

| **conditionId** | **operatorType** | **targetId** | **targetValue** |
|--------------|------------------|-------------|---------------|
| cond1        | setNetAssignment | net_input1  | X             |

Note, to ensure correct mapping, `setNetAssignment` must always be used for inputs when the neural network is part of the ODE RHS or observable formula.

### `operatorType` Defines Hybrid Model Type

The condition table and mapping table together specify where a neural network model is located in a PEtab SciML problem. In particular:

- If all inputs use `setNetAssignment` and all outputs use either `setNetRate` or `setNetAssignment`, the neural network appears in the ODE RHS.
- If all inputs use `setNetAssignment` and no outputs appear in the condition table, the neural network is part of the observable formula (note that the output variable must be referenced in the observable table).
- If no inputs use `setNetAssignment` and all outputs use either `setValue` or `setInitial`, the neural network sets model parameters or initial values before the simulation.

All other combinations are disallowed because they generally do not make sense in a PEtab context. For example, if inputs use `setNetAssignment` and outputs use `setValue`, parameter values prior to simulation would be set via an assignment rule consisting of model equations, which is not permitted in PEtab as assignment rules might be time-dependent. Moreover, if a parameter is to be set via an assignment rule, this should already be coded in the model. Implementations must ensure that the input combinations in the condition table are valid.

## Parameter Table

The parameter table follows the same format as in PEtab version 2, with a subset of fields extended to accommodate neural network parameters and new `initializationPriorType` values for neural network-specific initialization. A general overview of the parameter table is available in the PEtab documentation; here, the focus is on extensions relevant to the SciML extension.

### Detailed Field Description

- **parameterId [String]**: Identifies the neural network or a specific layer/parameter array. For example, `layerId` for `netId` can be specified using `netId.layerId`. A row for `netId` must be defined in the table. When parsing, more specific levels (e.g., `netId.layerId`) take precedence for nominal values, priors, etc.
- **nominalValue [String \| NUMERIC]**: Specifies neural network nominal values. This can be:
  - A path to an HDF5 file that follows PyTorch syntax (recommended, see above for file format). If no file exists when the problem is imported and the parameters are set to be estimated, a file is created with randomly sampled values.
  - A numeric value applied to all parameters under `netId`.
- **estimate [0 \| 1]**: Indicates whether the parameters are estimated (`1`) or fixed (`0`). This must be consistent across layers. For example, if `netId` has `estimate = 0`, then `netId.layerId` must also be `0`. In other words, freezing individual network parameters is not allowed.
- **initializationPriorType [String, OPTIONAL]**: Specifies the prior used for sampling initial values before parameter estimation. In addition to the PEtab-supported priors [ADD], the SciML extension supports the following standard neural network initialization priors:
  - [`kaimingUniform`](https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_uniform_) (default) — with `gain` as `initializationPriorParameters` value.
  - [`kaimingNormal`](https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_normal_) — with `gain` as `initializationPriorParameters` value.
  - [`xavierUniform`](https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.xavier_uniform_) — with `gain` as `initializationPriorParameters` value.
  - [`xavierNormal`](https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.xavier_normal_) — with `gain` as `initializationPriorParameters` value.

### Different Priors for Different Layers

Different layers can be defined for different layers. For example, consider a neural-network model `net1` where `layer1` and `layer2` should have different `initializationPriorParameters`, because they use different activation functions that should distinct [gain](https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.calculate_gain) for the `kaimingUniform` prior. A valid parameter table:

| **parameterId** | **parameterScale** | **lowerBound** | **upperBound** | **estimate** | **nominalValue** | **initializationPriorType** | **initializationPriorParameters** |
|---------------|------------------|--------------|--------------|------------|----------------|---------------------------|---------------------------------|
| net1          | lin              | -inf         | inf          | 1          | net1_ps.hf5    | kaimingUniform            | 1                               |
| net1.layer1   | lin              | -inf         | inf          | 1          | net1_ps.hf5    | kaimingUniform            | 1                               |
| net1.layer2   | lin              | -inf         | inf          | 1          | net1_ps.hf5    | kaimingUniform            | 5/3                             |

If is also possible to have different priors for different parameter arrays in a layer. For example, to use different priors for the weights and bias in `layer1` of `net1`, which is, for instance, a `linear` layer. In this case, a valid parameter table would be:

| **parameterId** | **parameterScale** | **lowerBound** | **upperBound** | **estimate** | **nominalValue** | **initializationPriorType** | **initializationPriorParameters** |
|--------------------|------------------|--------------|--------------|------------|----------------|---------------------------|---------------------------------|
| net1               | lin              | -inf         | inf          | 1          | net1_ps.hf5    | kaimingUniform            | 1                               |
| net1.layer1.weight | lin              | -inf         | inf          | 1          | net1_ps.hf5    | kaimingUniform            | 1                               |
| net1.layer1.bias   | lin              | -inf         | inf          | 1          | net1_ps.hf5    | kaimingNormal             | 5/3                             |

### Bounds for neural net parameters

Bounds can be specified for an entire network or its nested levels. However, it should be noted that most optimization algorithms used for neural networks, such as ADAM, do not support parameter bounds in their standard implementation.

## Problem YAML File

The PEtab problem YAML file follows the format of PEtab version 2, except that a mapping table is required (it is optional in the standard). It also includes an extension section to specify the neural network YAML files:

```yaml
extensions:
  petab_sciml:
    netId1:
      file: "file_path1.yaml"
    netId2:
      file: "file_path2.yaml"
```

Here, `netId1` and `netId2` are the IDs of the neural network models. Note that any number of neural networks can be specified. For example, for a model with one neural network, with ID `net1`, a valid YAML file would be:

```yaml
format_version: 2
problems:
  - model_files:
      model_sbml:
        location: "model.xml"
        language: "sbml"
    measurement_files:
      - "measurements.tsv"
    observable_files:
      - "observables.tsv"
    condition_files:
      - "conditions.tsv"
    mapping_files:
      - "mapping_table.tsv"
parameter_file: "parameters.tsv"
extensions:
  petab_sciml:
    net1:
      file: "net1.yaml"
```
