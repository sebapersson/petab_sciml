# Format Specification

A PEtab SciML problem extends the PEtab standard version 2 to accommodate hybrid models (SciML problems) that combine data-driven (neural net) and mechanistic components. The extension introduces one new PEtab file type:

1. **Neural Net File(s)**: Optional YAML file(s) describing neural net model(s).

It further extends the following standard PEtab files:

1. **Mapping Table**: Used to describe how neural network inputs and outputs map to PEtab quantities.
2. **Parameters Table**: Used to describe nominal values and potential priors for initializing network parameters.
3. **Condition Table**: Used to assign neural network outputs and inputs.
4. **Problem YAML File**: Includes a new SciML field for neural network models and (optionally) array or tensor formatted data.

All other PEtab files remain unchanged. This page explains the format and options for each file that is added or modified by the PEtab SciML extension.

## High Level Overview

The aim of the PEtab SciML extension is to facilitate the creation of hybrid models that combine data-driven and mechanistic components. The extension supports three hybrid model types, and a valid PEtab SciML problem can use one or any combination of these types. Each of the three types is specified differently:

1. **Data-driven models in the ODE model’s right-hand side (RHS):** In this scenario, the SBML file is modified during import by either replacing a derivative or setting a parameter value to a neural network output. In both cases, the neural network input and output variables (as defined in the mapping table) must be assigned in the condition table using the `setRate`, `setParameter`  and/or `setAssignment` operator types.

2. **Data-driven models in the observable function:** In this scenario, the neural network output variable (as defined in the mapping table) is directly inserted in the observable formula. Meanwhile, the input variables (also defined in the mapping table) are assigned in the condition table using the `setAssignment` operator type.

3. **Data-driven models to parametrize ODEs:** In this scenario, the data-driven model sets constant parameters or initial values in the ODE model prior to simulation. The input variable (as defined in the mapping table) can be assigned in the parameter or condition table as a standard constant PEtab variable, and the output variables (as defined in the mapping table) are assigned via the condition table.

## Neural Network Model Format

The neural network model format is flexible, meaning that data-driven models can be provided in any format supported by tools compatible with the PEtab SciML format (for example, [Lux.jl](https://github.com/LuxDL/Lux.jl) in [PEtab.jl](https://github.com/sebapersson/PEtab.jl)). Additionally, the `petab_sciml` library offers a YAML file format (see below) for neural network models, which can be imported into tools across programming languages. The reason for this flexibility in format is that, although the YAML format can accommodate many architectures, some may still be difficult to represent. Still, when possible, we recommend using the YAML format to facilitate model exchange across different software.

Regardless of the model format, to be compatible with the PEtab SciML format a neural network model must include two main parts:

- **layers**: A constructor that defines the network layers, each with a unique identifier.
- **forward**: A forward pass function that, given input arguments, specifies the order of layer calls and any activation functions used and returns an array output.

### YAML Network file format

The `petab_sciml` library provides a YAML file format for neural network model exchange, where layer names and argument syntax follow PyTorch conventions. Although the YAML files can be written manually, the recommended approach is to define a PyTorch `nn.Module`—using the constructor to set up the layers and the `forward` method to specify how they are invoked. For example, a simple feed-forward network can be defined as:

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
# TODO: Add when syntax is decided upon
```

Any PyTorch-supported keyword can be supplied for each layer in the YAML file, allowing for a broad range of architectures. For example, a more complex convolutional model could be created by:

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

### [Neural Network Parameters](@id hdf5_ps_structure)

All parameters for a neural network model are stored in an HDF5 file, with the file path specified in the problem [YAML file](@ref YAML_file). In this file, each layer’s parameters are in a group with identifier `f.layerId`, where `layerId` is the layer’s unique identifier. More formally, an HDF5 parameter file should have the following structure for an arbitrary number of layers:

```
parameters.hdf5
└───layer1 (group)
│   ├── arrayId{1}
│   └── arrayId{2}
└───layer2 (group)
    ├── arrayId{1}
    └── arrayId{2}
```

Here, `arrayId` depends on the naming convention for parameters in each layer. For example, a PyTorch `linear` layer typically has arrays named `weight` and (optionally) `bias`. While these names are common in many layers, the actual `arrayId` depends on the layer type and the specific neural network library.

Because parameters are stored in HDF5, they are saved as arrays. The indexing convention and naming therefore depends on the model library:

- For neural network models in the PEtab SciML YAML format, indexing follows PyTorch conventions. Usually, users do not need to handle these details directly, as PEtab SciML tools manage them automatically. This also means that `arrayId` follows PyTorch naming convention.
- For neural networks provided by another library, the indexing and `arrayId` naming follow that library’s conventions.

### [Neural Network Input](@id hdf5_input_structure)

When network input is provided as an array via an HDF5 file (see the mapping table below), its format should be:

```
input.hdf5
└───input (group)
    └─── input_array
```

As with [parameters](@ref hdf5_ps_structure), the indexing depends on the neural network library:

- For neural network models in the PEtab SciML format, indexing follows PyTorch conventions. For example, if the first layer is `Conv2d`, the input should be in `(C, W, H)` format, with data stored in row-major order. In general, the input should be structured to be directly compatible with PyTorch.
- For neural networks provided by another library, the indexing and ordering follow the conventions of that library.

!!! tip "For developers: Respect memory order"
    Tools supporting the SciML extension should, for computational efficiency, reorder input data and potential layer parameter arrays to match the memory ordering of the target language. For example, PEtab.jl converts input data to column-major order, with Julia indexing.

TODO: We will fix condition specific input in the YAML file later.

## Mapping Table

To avoid confusion regarding what a neural network ID (`netId`) refers to (e.g., parameters, inputs, etc.), `netId` is not considered a valid PEtab identifier. Consequently, every neural network input, parameter, and output must be explicitly mapped in the mapping table to a PEtab variable. In the context of the PEtab SciML extension, the relevant mapping table columns are:

- **petabEntityId [STRING]**: A valid PEtab identifier that is not defined elsewhere in the PEtab problem. This identifier can be referenced in the condition, measurement, parameter, and observable tables, but not within the model itself. For neural network outputs, the PEtab identifier must be assigned in the condition table; for inputs, this is not required (see examples below).
- **modelEntityId [STRING]**: Describes the neural network entity corresponding to the `petabEntityId`. This must specify a parameter set (e.g. `netId.parameters`), a parameter for a specific layer (e.g. `netId.parameters.layerId`) an input (e.g. `netId.input[{n}]`), or an output (`netId.output[{n}]`), where `n` is the specific input or output index.

### Network with Scalar Inputs

For networks with scalar inputs, the PEtab entity should be a PEtab variable. For example, assume that the network `net1` has two inputs, then a valid mapping table would be:

| **petabEntityId** | **modelEntityId** |
|-------------------|-------------------|
| net1\_input1       | net1.input[1]       |
| net1\_input2       | net1.input[2]       |

Scalar input variables can then be:

- **Parameters in the parameters table**: These may be either estimated or constant.
- **Parameters assigned in the condition table**: More details on this option can be found in the section on the condition table.

### Network with Array Inputs

Sometimes, such as with image data, a neural network requires array input. In these situations, the input should be specified as an HDF5 file (for file structure [here](@ref hdf5_input_structure)), which is given a suitable PEtab Id in the [YAML file](@ref YAML_file). This Id is then mapped to a PEtab variable in the mapping table. For example, given that `net1` has a file input, a valid mapping table would be:

| **petabEntityId** | **modelEntityId** |
|-------------------|-------------------|
| net1_input_file   | net1.input        |

Where `net1_input_file` has been specified in the problem [YAML file](@ref YAML_file).

When multiple simulation conditions each require a different neural network array input, the mapping table should map the input to a PEtab variable (for example, `net1\_input`):

| **petabEntityId** | **modelEntityId** |
|-------------------|-------------------|
| net1\_input        | net1.input        |

This variable (here `net1_input`) is then assigned to specific input file variables (e.g. `net1\_input_cond1` and `net1\_input_cond2`) via the condition table using the `setValue` operator type. For a full example of a valid PEtab problem with array inputs, see [ADD].

TODO: Will fix an input format later.

### Network with Multiple Input Arguments

Sometimes a neural network model’s forward function has multiple input arguments. When the `netId.input[{n}]` notation is used in the mapping table it is assumed that there is only one input argument. Therefore, when there are multiple input arguments, the `netId.inputs[{n}][{m}]` notation should be used. For example, if there are two input arguments, each taking a scalar value, a valid mapping table would be:

| **petabEntityId** | **modelEntityId**    |
|-------------------|----------------------|
| net1\_arg1         | net1.inputs[1][1]    |
| net1\_arg2         | net1.inputs[2][1]    |

### [Network Observable Formula Output](@id output_obs)

If the neural network output appears in the observable formula, the PEtab entity should be directly referenced in the observable formula. For example, given:

| **petabEntityId** | **modelEntityId**  |
|-----------------|-----------------|
| net1\_output1    | net1.output[1]    |

A valid entry in the observable table would be:

| **observableId** | **observableFormula** |
|----------------|----------------------|
| obs1           | net1\_output1        |

As usual, the `observableFormula` can be any valid PEtab equation, so `net1_output1 + 1` would also be valid.

### Network Scalar Output

If the output does not appear in the observable formula, the output variable should still be defined in the mapping table:

| **petabEntityId** | **modelEntityId**  |
|-----------------|-----------------|
| net1\_output1    | net1.output[1]    |

The output parameter (`net1_output1`) is then assigned in the condition table (see below).

### [Network Parameter Values](@id mapping_ps)

The PEtab ID representing the parameters for a neural network model must also be assigned in the mapping table. For example, if the network is called `net1`, a valid mapping table entry would be:

| **petabEntityId** | **modelEntityId**   |
|-------------------|---------------------|
| net1_ps           | net1.parameters     |

Next, `net1_ps` should be assigned properties in the [parameter table](@ref parameter_table).

It is also possible to target a specific layer in the parameter table. To do so, the layer must first be mapped to a PEtab variable. For the case above, a valid mapping table to target `layer1` would be:

| **petabEntityId** | **modelEntityId**         |
|-------------------|---------------------------|
| net1\_ps           | net1.parameters          |
| net1\_ps\_layer1    | net1.parameters.layer1   |

It is also possible to target parameter arrays within a layer using the notation `netId.layerId.arrayId`.

### Additional Details

Although a neural network can, in principle, accept both array and scalar inputs for a single argument, this feature is not currently tested for among tools implementing the PEtab SciML extension due to it being hard to implement. To have both scalar and array input, the neural network model should instead have multiple arguments for its forward pass function.

## Condition Table

In the PEtab SciML extension, the condition table is expanded to specify how neural network outputs—and, if necessary, inputs—are assigned. Additionally, a new `operatorType` called `setParameter` is introduced to support inserting neural networks into the ODE right-hand side (RHS). Repeating parts of the PEtab v2 standard, the three key `operatorType` values for this extension are:

1. **setRate**: Assigns the rate of a species to a neural network output. Here, `targetValue` must be a neural network output, and `targetId` must be a model species.
2. **setAssignment**: Assigns the input of a neural network in the ODE RHS or the input in the observable formula. Note, neural network inputs are treated as algebraic targets.
3. **setParameter**: Assigns the value of a model parameter to a neural network output. Here, `targetValue` must be a neural network output.

!!! warn "Model structure altering conditions"
    When `setRate` or `setParameter` are used, during model import the generated model structure or observable formula is altered, basically a neural-network is inserted into the generated functions. Therefore, as unique model structures per condition are not supported in most PEtab tools, the same `setAssignment` or `setRate` assignment must be set per condition.

### `operatorType` Defines Hybrid Model Type

The condition table and mapping table together specify where a neural network model is located in a PEtab SciML problem. In particular:

- If all inputs use `setAssignment` and all outputs use either `setRate` or `setParameter`, the neural network appears in the ODE RHS.
- If all inputs use `setAssignment` and no outputs appear in the condition table, the neural network is part of the observable formula (note that the output variable must be referenced in the observable table).
- If no inputs use `setAssignment` and all outputs use either `setValue` or `setInitial`, the neural network sets model parameters or initial values before the simulation.

All other combinations are disallowed because they generally do not make sense in a PEtab context. For example, if inputs use `setAssignment` and outputs use `setValue`, then parameter values prior to simulation would be set by an assignment rule derived from model equations, which is not allowed in PEtab because assignment rules might be time-dependent. Furthermore, if a parameter is to be assigned via a rule, it should already be incorporated into the model. The `petab_sciml` library provides a linter to ensure no disallowed combination is used. Packages that do not wish to depend on Python are strongly encouraged to verify that the input combinations in the condition table are valid.

### Assigning Neural Network Output

To set a **constant model parameter value** before model simulations, the `setValue` operator type should be used. For example, if parameter `p` is determined by `net1_output1` (mapped to a neural network output in the mapping table), a valid condition table entry is:

| **conditionId** | **operatorType** | **targetId** | **targetValue** |
|-----------------|------------------|--------------|-----------------|
| cond1           | setValue         | p            | net1\_output1    |

Note that this specification allows for condition-specific assignments. For example, `net1_output1` could target a different parameter in another condition or multiple model parameters in the same condition.

To set an **initial value**, the `setInitial` operator type should be used. For example, if the initial value of species `X` comes from `net1_output1`, a valid condition table entry is:

| **conditionId** | **operatorType** | **targetId** | **targetValue** |
|-----------------|------------------|--------------|-----------------|
| cond1           | setInitial       | X            | net1_output1    |

To set a **model derivative**, the `setRate` operator type should be used. For example, if the rate of species `X` is given by `net1_output1`, a valid condition table entry is:

| **conditionId** | **operatorType** | **targetId** | **targetValue** |
|-----------------|------------------|--------------|-----------------|
| cond1           | setRate       | X            | net1\_output1    |

To alter the **ODE RHS**, the `setParameter` operator type should be used. For example, if an ODE model parameter `p` should be given by `net1_output1`, a valid condition table entry is:

| **conditionId** | **operatorType**    | **targetId** | **targetValue** |
|-----------------|---------------------|--------------|-----------------|
| cond1           | setParameter    | p            | net1\_output1    |

In theory, the `targetId` (here `p`) can also appear in the observable formula. However, as described [here](@ref output_obs), for the observable formula the mapped output variable must be directly encoded in the formula. Therefore, `targetId` appearing in the observable formula is not allowed, and the `petab_sciml` linter checks for this.

### Assigning Neural Network Input

When a neural network sets a **constant model parameter value or initial value**, its input variable (as specified in the mapping table) is a standard PEtab variable. If that input variable is not defined in the parameter table, it should be assigned using the `setValue` operator type.

When a neural network sets **a model derivative or alters the ODE RHS**, the input typically depends on model entities. Therefore, the input variable should be assigned using the `setAssignment` operator. For example, if neural network input `net1_input1` is given by specie `X`, a valid condition table is:

| **conditionId** | **operatorType** | **targetId** | **targetValue** |
|--------------|------------------|-------------|---------------|
| cond1        | setAssignment | net_input1  | X             |

Note, to ensure correct mapping, `setAssignment` must always be used for inputs when the neural network is part of the ODE RHS or observable formula.

## [Parameter Table](@id parameter_table)

The parameter table follows the same format as in PEtab version 2, with a subset of fields extended to accommodate neural network parameters and new `initializationPriorType` values for neural network-specific initialization. A general overview of the parameter table is available in the PEtab documentation; here, the focus is on extensions relevant to the SciML extension.

!!! note "Specific Assignments Have Precedence"
    When parsing, more specific assignments (e.g., `netId.layerId` rather than `netId`) take precedence for nominal values, priors, and so on. This means that if `netId` has one prior, and `netId.layerId` has another, the more specific assignment `netId.layerId` overrides the less specific one `netId` for `layerId` in this case.

### Detailed Field Description

- **parameterId [String]**: Identifies the neural network or a specific layer/parameter array. The target of the `parameterId` must be assigned via the [mapping table](@ref mapping_ps).
- **nominalValue [String \| NUMERIC]**: Specifies neural network nominal values. This can be:
  - A PEtab variable that via the problem [YAML file](@ref YAML_file) maps to an HDF5 file with the required [structure](@ref hdf5_ps_structure). If no file exists at the given path when the problem is imported and the parameters are set to be estimated, a file is created with randomly sampled values.
  - A numeric value applied to all parameters under `parameterId`.
- **estimate [0 \| 1]**: Indicates whether the parameters are estimated (`1`) or fixed (`0`). This must be consistent across layers. For example, if `netId` has `estimate = 0`, then potential layer rows must also be `0`. In other words, freezing individual network parameters is not allowed.
- **initializationPriorType [String, OPTIONAL]**: Specifies the prior used for sampling initial values before parameter estimation. In addition to the PEtab-supported priors [ADD], the SciML extension supports the following standard neural network initialization priors:
  - [`kaimingUniform`](https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_uniform_) (default) — with `gain` as `initializationPriorParameters` value.
  - [`kaimingNormal`](https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_normal_) — with `gain` as `initializationPriorParameters` value.
  - [`xavierUniform`](https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.xavier_uniform_) — with `gain` as `initializationPriorParameters` value.
  - [`xavierNormal`](https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.xavier_normal_) — with `gain` as `initializationPriorParameters` value.

### Different Priors for Different Layers

Different layers can have distinct initialization prior parameters. For example, consider a neural-network model `net1` where `layer1` and `layer2` require different `initializationPriorParameters` because they use different activation functions that need distinct [gain](https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.calculate_gain) values for the `kaimingUniform` prior. A valid parameter table would be:

| **parameterId**    | **parameterScale** | **lowerBound** | **upperBound** | **estimate** | **nominalValue** | **initializationPriorType** | **initializationPriorParameters** |
|--------------------|--------------------|----------------|----------------|------------|------------------|-----------------------------|-------------------------------------|
| net1\_ps            | lin                | -inf           | inf            | 1          | net1\_ps\_file      | kaimingUniform              | 1                                   |
| net1\_layer1_ps     | lin                | -inf           | inf            | 1          | net1\_ps\_file      | kaimingUniform              | 1                                   |
| net1\_layer2_ps     | lin                | -inf           | inf            | 1          | net1\_ps\_file      | kaimingUniform              | 5/3                                 |

Where `parameterId` are assumed to have been properly assigned in the [mapping table](@ref mapping_ps). In this example, each layer references the same file variable for `nominalValue`. This means the layers obtain their values from the specified file. Unless a numeric value is provided for `nominalValue`, referring to the same file is required, since all neural network parameters should be collected in a single HDF5 file following the structure described [here](@ref hdf5_ps_structure).

It is also possible to specify different priors for different parameter arrays within a layer. For example, to use different priors for the weights and bias in `layer1` of `net1` (assuming a `linear` layer), a valid parameter table would be:

| **parameterId**      | **parameterScale** | **lowerBound** | **upperBound** | **estimate** | **nominalValue** | **initializationPriorType** | **initializationPriorParameters** |
|----------------------|--------------------|----------------|----------------|------------|------------------|-----------------------------|-------------------------------------|
| net1\_ps              | lin                | -inf           | inf            | 1          | net1\_ps\_file      | kaimingUniform              | 1                                   |
| net1\_layer1_weight   | lin                | -inf           | inf            | 1          | net1\_ps\_file      | kaimingUniform              | 1                                   |
| net1\_layer1_bias     | lin                | -inf           | inf            | 1          | net1\_ps\_file      | kaimingNormal               | 5/3                                 |

### Bounds for neural net parameters

Bounds can be specified for an entire network or its nested levels. However, it should be noted that most optimization algorithms used for neural networks, such as ADAM, do not support parameter bounds in their standard implementation.

## [Problem YAML File](@id YAML_file)

The PEtab problem YAML file follows the PEtab version 2 format, except that a mapping table is now required (it is optional in the standard). It also includes an extension section for specifying neural network YAML files, as well as any files for neural network parameters and/or inputs:

```yaml
extensions:
  petab_sciml:
    netId1:
      location: file_path1.yaml
    netId2:
      location: file_path2.yaml
    array_files:
      netId1_input:
        location: netId1_input.hdf5
        language: hdf5
      netId1_ps:
        location: netId1_ps.hdf5
        language: hdf5
```

Here, `netId1` and `netId2` are the IDs of the neural network models, and `net1_input` and `net1_ps` corresponding to array files that can be used in the PEtab tables. In this case, `net1_input` would be used in the mapping table for `netId1`’s input, while `netId1_ps` would be the `nominalValue` in the parameter table.

If the neural network is provided in another format—typically one specific to a certain implementation—the neural network library should be provided to inform users which library is used. For example, when using Lux.jl in Julia, a valid file would be:

```yaml
extensions:
  petab_sciml:
    netId1:
      library: Lux.jl
    netId2:
      library: Lux.jl
```

It is then up to the specific implementation to provide the neural network model during import of the PEtab problem.

Any number of neural networks can be specified. For example, for a model with a single neural network with ID `net1`, a valid YAML file would be:

```yaml
format_version: 2
problems:
  - model_files:
      model_sbml:
        location: model.xml
        language: sbml
    measurement_files:
      - measurements.tsv
    observable_files:
      - observables.tsv
    condition_files:
      - conditions.tsv
    mapping_files:
      - mapping_table.tsv
parameter_file: parameters.tsv
extensions:
  petab_sciml:
    net1:
      location: net1.yaml
    array_files:
      net1_ps:
        location: net1_ps.hdf5
        language: hdf5
```
