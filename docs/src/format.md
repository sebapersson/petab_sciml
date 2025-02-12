# Format Specification

A PEtab SciML problem extends a standard version 2 PEtab problem to accommodate hybrid models SciML combining data-driven and mechanistic components. To this, the extension introduces one new file:

1. **Neural Net Fil**: YAML file(s) describing neural net model(s).

It also extends the following standard PEtab files to accommodate SciML models:

1. **Mapping Table**: Describes how neural network inputs and outputs map to PEtab quantities.
2. **Parameters Table**: Describes values and potential priors for initializing network parameters.
3. **Condition Table**: Allows assigning neural network outputs and inputs.
4. **Problem YAML File**: Includes a SciML field.

All other PEtab files remain unchanged. 

The main goal of the PEtab SciML extension is to enable hybrid models that combine data-driven and mechanistic components. There are three such hybrid model types, each specified differently:

1. **Data-driven models in the ODE model’s right-hand side (RHS)**
   - During import, the SBML file is altered by either replacing a derivative or assigning a parameter to a neural network input.
   - In these cases, both the neural network inputs and outputs (as defined in the mapping table) must be assigned in the condition table with the `setNetRate` and/or `setNetAssignment` `operatorType`.
2. **Data-driven models in the observable function**
   - The output variable (as defined in the mapping table) is code directly in the observable formula.
   - The input variables (as defined in the mapping table) are assigned in the condition table with the `setNetAssignment` `operatorType`.
3. **Data-driven models before the ODE model**
   - These models set constant parameters or initial values in the ODE model prior to simulations.
   - The input can be defined in the mapping table or in the condition table.
   - The output variables (as defined in the mapping table) are assigned via the condition table.

This page explains each file that is added or modified by the PEtab SciML extension, highlights its main functionality, and outlines the import logic for each supported hybrid model type.

## Neural Network Model Format

**TODO:** Dilan, I will need some help from you here, link the scheme?

The neural network models are provided as separate YAML files that can be imported into a chosen neural-network package. Each YAML file has two main sections:

- **`layers`**: Defines the neural network layers, each with a unique ID. The layer names and argument syntax follow PyTorch conventions.
- **`forward`**: Describes the forward pass, indicating the order of layer calls and any activation functions.

Although these YAML files can be written manually, the recommended approach is to define a PyTorch `nn.Module` whose constructor sets up the layers and whose `forward` method outlines the layer calls. For example, a simple feed-forward network file can be created via:

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

Where the YAML file can then be generated with the `petab_sciml` library:

```python
# TODO: Add
```

Any PyTorch-supported keyword can be supplied for each layer in the YAML file, and a wide range of layers are available. For instance, a more complex convolutional model might look like:

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

A complete list of supported and tested layers and activation functions can be found [ADD].

### Neural Network Parameters

A subset of supported layers have parameters. In the PEtab SciML extension, all parameters for a neural network model are stored in an HDF5 file, with the file name specified in the parameter table. Each layer’s parameters are grouped under `f.layerId`, where `layerId` is the layer’s unique identifier. For example, the weights of a linear layer are stored at `f.layerId.weight`.

Since parameters are stored in an HDF5 format, they are stored as arrays. The indexing follows PyTorch conventions, meaning parameters are stored in row-major order. Typically, users do not need to manage these details manually, as PEtab SciML tools handle them automatically.

### Neural Network Input

When network input is provided as an array via an HDF5 file (see the mapping table below), it should follow the PyTorch convention. For example, if the first layer is `Conv2d`, the input should be in `(C, W, H)` format, with data stored in row-major order. In general, the input should be structured to be directly compatible with PyTorch.

!!! tip "For developers: Respect memory order"
    Tools supporting the SciML extension should, for computational efficiency, reorder input data and potential layer parameter arrays to match the memory ordering of the target language. For example, PEtab.jl converts input data to column-major order, as used in Julia.

## Mapping Table

The mapping table describes how a neural network’s inputs and outputs map to PEtab problem variables.

| **petabEntityId** | **modelEntityId** |
|-------------------|-------------------|
| e.g.              |                   |
| k1                | netId.input1      |

### Detailed Field Descriptions

- **`petabEntityId` [STRING]**: A valid PEtab identifier not defined elsewhere in the PEtab problem. It can be referenced in the condition, measurement, parameter, and observable tables or be a file, but not in the model itself. For neural network outputs, the PEtab identifier must be assigned in the condition table, whereas for inputs, this is not required (see examples below).
- **`modelEntityId` [STRING]**: Describes the neural network entity corresponding to the `petabEntityId`. Must follow the format `netId.input{n}` or `netId.output{n}`, where `n` is the specific input or output index.

### Example: Network with Scalar Inputs

Assume that the network `net1` has two inputs, PEtab problem parameters `net1_input1` and `net1_input2`. This would be specified as:

| `petabEntityId` | `modelEntityId`  |
|-----------------|------------------|
| net1_input1              | net1.input1      |
| net1_input2              | net1.input2      |

In particular, scalar inputs can be:

- **Parameters in the parameters table**: These may be either estimated or constant.
- **Assigned in the condition table**: This allows for condition-specific neural network inputs. Additionally, inputs can via the condition table be defined as equations, which is useful when the neural network is part of an ODE right-hand side.

### Example: Network with Array Inputs

Sometimes, such as with image data, a neural network requires array input. In these cases, the input can be specified as an HDF5 file directly in the mapping table:

| `petabEntityId`   | `modelEntityId`  |
|-------------------|------------------|
| input_net1.hf5    | net1.input1      |

As mentioned in [ADD], the HDF5 file should follow PyTorch indexing and be stored in row-major order.

When there are multiple simulation conditions that each require a different neural network array input, the mapping table should map to a PEtab variable (e.g., `net1_input`):

| `petabEntityId` | `modelEntityId`  |
|-----------------|------------------|
| net1_input      | net1.input1      |

This variable is then assigned to specific input files via the condition table. For a full example of a valid PEtab problem with array inputs, see [ADD].

### Example: Network Observable Formula Output

If the neural network output appears in the observable formula, the PEtab entity should be directly referenced in the formula. For example:

| `petabEntityId` | `modelEntityId` |
|-----------------|-----------------|
| net1_output1    | net1.output1    |

A valid observable table would then be:

| `observableId` | `observableFormula` |
|----------------|----------------------|
| obs1           | net1_output1        |

As usual, the `observableFormula` can be any valid PEtab equation, so `net1_output1 + 1` would also be valid.

### Example: Network Scalar Output

In addition to the observable formula, a neural network output can set a constant model parameter or be used in the ODE model right-hand side. In this case, the output is mapped to a PEtab parameter. For example:

| `petabEntityId` | `modelEntityId` |
|-----------------|------------------|
| net1_output1    | net1.output1     |

The parameter (`net1_output1`) is then assigned in the condition table (see below), allowing anything from model parameters to derivatives to be set.

### Additional Details

Although a neural network can, in principle, accept both array and scalar inputs, this feature is not currently tested for among tools implementing the PEtab SciML extension due to it being hard to implement. However, tools are free to add this feature.

## Condition Table

In the PEtab SciML extension, the condition table is extended to specify how neural network outputs (and, if needed, inputs) are assigned. To support this, two new operator types are introduced in `operatorType`:

1. **`setNetRate`**: Assigns the rate of a species to a neural network output. In this case, `targetValue` must be a neural network output.
2. **`setNetAssignment`**: Assigns the input or output of a neural network in the ODE or observable formula.
   - **Input Case**: `targetId` is a neural network input, and `targetValue` can be any valid math expression consisting of model variables.
   - **Output Case**: `targetId` is an ODE model parameter, and `targetValue` is a non-estimated model parameter that is replaced by the neural network output. Used to assign output in the ODE model RHS.

### Examples: Specifying Neural Network Output

#### Set a Constant Model Parameter

Use the `setValue` operator. For example, if parameter `p` is determined by `net1_output1` (mapped to a neural network output in the mapping table), write:

| `conditionId` | `operatorType` | `targetId` | `targetValue` |
|---------------|----------------|------------|---------------|
| cond1         | setValue       | p          | net1_output1  |

This approach allows for condition-specific assignments. For example, `net1_output1` could target a different parameter in another condition or multiple parameters in the same condition.

#### Set an Initial Value

Use the `setInitial` operator. For example, if the initial value of species `X` comes from `net1_output1`, write:

| `conditionId` | `operatorType` | `targetId` | `targetValue` |
|---------------|----------------|------------|---------------|
| cond1         | setInitial     | X          | net1_output1  |

#### Set a Model Derivative

Use the `setNetRate` operator. For example, if the derivative of `X` should be assigned to `net1_output1`, write:

| `conditionId` | `operatorType` | `targetId` | `targetValue` |
|---------------|----------------|------------|---------------|
| cond1         | setNetRate     | X          | net1_output1  |

#### Alter the ODE RHS

Use the `setNetAssignment` operator. For example, if an ODE model parameter `p` should be given by `net1_output1`, write:

| `conditionId` | `operatorType` | `targetId` | `targetValue` |
|---------------|----------------|------------|---------------|
| cond1         | setNetAssignment | p          | net1_output1  |

### Specifying Neural Network Input

#### Neural Network Sets Initial Value(s) or Constant Model Parameter(s)

When a neural network sets a constant model parameter value or an initial input, the input should be defined as a PEtab entity and specified directly in the mapping table (see above). This input can then be assigned in the parameters table or the conditions table using the `setValue` operator.

#### Neural Network in the ODE RHS or Observable Formula

If the neural network appears in the ODE right-hand side or the observable formula, its input typically depends on model entities. Therefore, the input variable should be assigned using the `setNetAssignment` operator. For example, suppose `net_input1` (mapped as a neural network input in the mapping table) should be assigned to species `X`:

| `conditionId` | `operatorType`    | `targetId`  | `targetValue` |
|--------------|------------------|-------------|---------------|
| cond1        | setNetAssignment | net_input1  | X             |

In some cases, a neural network input may not correspond to a model variable (e.g., it could be a constant scalar). However, to ensure correct mapping, `setNetAssignment` must always be used for inputs when the neural network is part of the ODE RHS or observable formula.

### Additional Details

The condition table and mapping table together specify where a neural network model is located in a PEtab SciML problem. In particular:

- If all inputs use `setNetAssignment` and all outputs use either `setNetRate` or `setNetAssignment`, the neural network appears in the ODE RHS.
- If all inputs use `setNetAssignment` and no outputs appear in the condition table, the neural network is part of the observable formula (note though that the output variable must be referenced in the observable table).
- If no inputs use `setNetAssignment` and all outputs use either `setValue` or `setInitial`, the neural network sets model parameters or initial values before the simulation.

All other combinations are disallowed because they generally do not make sense in a PEtab context. For example, if inputs use `setNetAssignment` and outputs use `setValue`, the parameter values prior to simulation would be set via an assignment rule, which is not permitted in PEtab as, with other things, assignment rules might be time dependent. Naturally, implementations need to test that input combinations in the condition table are valid.

## Parameters Table

The parameter table follows the same format as in PEtab version 2 but extends it to accommodate neural network parameters and introduces new `initializationPriorType` for neural network-specific initialization. A general overview of the parameter table is available in the PEtab documentation; here, the focus is on extensions relevant to SciML extension.

### Detailed Field Descriptions (Neural Network Extension)

- **`parameterId` [String]**: Identifies the neural network or a specific layer/parameter array. For example, `layerId` for `netId` can be specified using `netId.layerId`. A row for `netId` must be defined in the table. When parsing, more specific parameters (e.g., `netId.layerId`) take precedence for nominal values, priors, etc.
- **`nominalValue` [String \| NUMERIC]**  Specifies neural network nominal values. This can be:
  - A path to an HDF5 file that follows PyTorch syntax (recommended, see above for file format). If no file exists when the problem is imported and the parameters are set to be estimated, a file is created with randomly sampled values.
  - A numeric value applied to all parameters under `netId`.
- **`estimate` [0 \| 1]**: Indicates whether the parameters are estimated (`1`) or fixed (`0`). This must be consistent across layers. For example, if `netId` has `estimate = 0`, then `netId.layerId` must also be `0`. In other words, freezing individual network parameters is not allowed.
- **`initializationPriorType` [String, OPTIONAL]**: Specifies the prior used for sampling initial values before parameter estimation. In addition to the PEtab-supported priors [ADD], the SciML extension supports the following standard neural network initialization priors:
  - [`kaimingUniform`](https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_uniform_) (default) — with `gain` as `initializationPriorParameters` value.
  - [`kaimingNormal`](https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_normal_) — with `gain` as `initializationPriorParameters` value.
  - [`xavierUniform`](https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.xavier_uniform_) — with `gain` as `initializationPriorParameters` value.
  - [`xavierNormal`](https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.xavier_normal_) — with `gain` as `initializationPriorParameters` value.

### Example: Different Priors for Different Layers

Consider a neural-network model `net1` where we want different `initializationPriorParameters` for `layer1` and `layer2`, because they use different activation functions that should distinct [gain](https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.calculate_gain) for the `kaimingUniform` prior. A valid parameter table would be:

| `parameterId` | `parameterScale` | `lowerBound` | `upperBound` | `estimate` | `nominalValue` | `initializationPriorType` | `initializationPriorParameters` |
|---------------|------------------|--------------|--------------|------------|----------------|---------------------------|---------------------------------|
| net1          | lin              | -inf         | inf          | 1          | net1_ps.hf5    | kaimingUniform            | 1                               |
| net1.layer1   | lin              | -inf         | inf          | 1          | net1_ps.hf5    | kaimingUniform            | 1                               |
| net1.layer2   | lin              | -inf         | inf          | 1          | net1_ps.hf5    | kaimingUniform            | 5/3                             |

### Example: Different Priors for Parameters in a Layer

Now consider we want to use different priors for the weights and bias in `layer1` of `net1`, which is, for instance, a `linear` layer. A valid parameter table would be:

| `parameterId`      | `parameterScale` | `lowerBound` | `upperBound` | `estimate` | `nominalValue` | `initializationPriorType` | `initializationPriorParameters` |
|--------------------|------------------|--------------|--------------|------------|----------------|---------------------------|---------------------------------|
| net1               | lin              | -inf         | inf          | 1          | net1_ps.hf5    | kaimingUniform            | 1                               |
| net1.layer1.weight | lin              | -inf         | inf          | 1          | net1_ps.hf5    | kaimingUniform            | 1                               |
| net1.layer1.bias   | lin              | -inf         | inf          | 1          | net1_ps.hf5    | kaimingNormal             | 5/3                             |

### Additional Details

It is not possible to specify parameters at a level deeper than arrays. For example, in a `linear` layer, the deepest allowed specification is `netId.layerId.weight`.

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

Here, `netId1` and `netId2` are the IDs of the neural network models. You can include any number of neural networks under `petab_sciml`.

### Example

For a model with one neural network, with ID `net1`, a valid YAML file would be:

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
