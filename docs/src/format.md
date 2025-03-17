# Format Specification

A PEtab SciML problem extends the PEtab standard version 2 to accommodate hybrid models (SciML problems) that combine neural network and mechanistic components. The extension introduces two new PEtab file types:

1. [Neural Net File(s)](@ref net_format): Optional YAML file(s) describing neural net model(s).
2. [Hybridization table](@ref hybrid_table): Table used to assign neural network outputs and inputs.

It further extends the following standard PEtab files:

1. [Mapping Table](@ref mapping_table): Extended to describe how neural network inputs and outputs map to PEtab variables.
2. [Parameters Table](@ref parameter_table): Extended to describe nominal values and potential priors for initializing network parameters.
3. [Problem YAML File](@ref YAML_file): Extended to include a new SciML field for neural network models and (optionally) array or tensor formatted data.

All other PEtab files remain unchanged. This page explains the format and options for each file that is added or modified by the PEtab SciML extension.

## High Level Overview

The aim of the PEtab SciML extension is to facilitate the creation of hybrid models that combine neural network and mechanistic components. The extension is designed to keep the dynamic model, neural network model, and PEtab problem as independent as possible, with the models linked in the hybridization and/or condition tables.

The extension supports three hybrid model types, and a valid PEtab SciML problem can use one or any combination of these types. The three types are:

1. **Neural network models in the ODE model’s right-hand side (RHS):** In this scenario, the model (e.g. SBML) file is modified during import by either replacing a derivative or setting a parameter value to a neural network output.
2. **Neural network models in the observable function:** In this scenario, the neural network output variable (as defined in the mapping table) is directly inserted in the observable formula.
3. **Neural network models to parametrize ODEs:** In this scenario, the neural network model sets constant parameters or initial values in the ODE model prior to simulation.

## [Neural Network Model Format](@id net_format)

The neural network model format is flexible, meaning that models can be provided in any format supported by tools compatible with the PEtab SciML format (for example, [Lux.jl](https://github.com/LuxDL/Lux.jl) in [PEtab.jl](https://github.com/sebapersson/PEtab.jl)). Additionally, the `petab_sciml` library offers a neural network YAML file format, which can be imported into tools across programming languages. This format flexibility exists because, although the YAML format can accommodate many architectures, some may still be difficult to represent. However, when possible, we recommend using the YAML format to facilitate model exchange across different software.

Regardless of the model format, to be compatible with the other files in a PEtab SciML problem a neural network model must consist of two parts:

- **layers**: A constructor that defines the network layers, each with a unique identifier.
- **forward**: A forward pass function that, given input arguments, specifies the order in which layers are called, applies any activation functions, and returns a single array as output. The forward function can accept more than one input argument (`n > 1`), and in the [mapping table](@ref mapping_table), the forward function's `n`th input argument (ignoring any potential class arguments such as `self`) is considered as argument `n`.

### [YAML Network file format](@id YAML_net_format)

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
        self.fc4 = nn.Linear(10, 10)
        self.flatten1 = nn.Flatten()

    def forward(self, input1, input2):
        # input1
        c1 = self.conv1(input1)
        s2 = self.max_pool1(c1)
        c3 = self.conv2(s2)
        s4 = self.max_pool1(c3)
        s4 = self.flatten1(s4)
        f5 = self.fc1(s4)
        f6 = self.fc2(f5)
        f8 = self.fc3(f6)
        # input 2
        f9 = self.fc4(input2)
        output = f8 + f9
        return output
```

Where in this case, the forward function has two inputs arguments. A complete list of supported and tested layers and activation functions can be found [here](@ref layers_activation).

### [Neural Network Parameters](@id hdf5_ps_structure)

All parameters for a neural network model are stored in an HDF5 file, with the file path specified in the problem [YAML file](@ref YAML_file). In this file, each layer’s parameters are in a group with identifier `f.layerId`, where `layerId` is the layer’s unique identifier. More formally, an HDF5 parameter file should have the following structure for an arbitrary number of layers:

```hdf5
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

```hdf5
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

## [Mapping Table](@id mapping_table)

To avoid confusion regarding what a neural network Id (`netId`) refers to (e.g., parameters, inputs, etc.), `netId` is not considered a valid PEtab identifier. Consequently, every neural network input, parameter, and output must be explicitly mapped in the mapping table to a PEtab variable.

### Detailed Field Description

- **petabEntityId** [STRING, required]: A valid PEtab identifier that is not defined elsewhere in the PEtab problem. This identifier can be referenced in the condition, hybridization, measurement, parameter, yaml file (in case of file input), and observable tables, but not within the model itself.
- **modelEntityId** [STRING, required]: Describes the neural network component mapped to the `petabEntityId`. For a neural network, the valid identifiers are:  
  - `netId.parameters`: Parameters for a neural network model. A specific layer can also be referenced with `netId.parameters.layerId`, and specific arrays in a layer can be referenced with `netId.parameters.arrayId`. For parameter arrays, individual indexing is not allowed.
  - `netId.input`: Input for a neural network with a single input argument (as in the first example [here](@ref YAML_net_format)). If the input is provided as an array from a file, use `netId.input`. Otherwise, the input is assumed to be a `Vector`, where each element `n` should be mapped to a PEtab ID, with element `n` specified as `netId.input[{n}]`.
  - `netId.inputs`: Inputs for a neural network with multiple input arguments (as in the second example [here](@ref YAML_net_format)). Each input argument is accessed via `netId.inputs[{n}]`, and for each argument the same rules apply as for `netId.input`. For example, to access element `m` in input `n`, write `netId.inputs[{n}][{m}]`.
  - `netId.output`: Neural network output, assumed to be a `Vector`, where each element `n` should be mapped to a PEtab ID, with element `n` given by `netId.input[{n}]`.

### Example: Network with Scalar Inputs

For networks with scalar inputs, the PEtab entity should be a PEtab variable. For example, assume that the network `net1` has two inputs, then a valid mapping table would be:

| **petabEntityId** | **modelEntityId** |
|-------------------|-------------------|
| net1\_input1       | net1.input[1]       |
| net1\_input2       | net1.input[2]       |

Scalar input variables can then be:

- **Parameters in the parameters table**: These may be either estimated or constant.
- **Parameters assigned in the condition or hybridization table**: More details on this option can be found in the section on the condition and hybridization tables.

### Example: Network with Array Inputs

Sometimes, such as with image data, a neural network requires array input. In these situations, the input should be specified as an HDF5 file (for file structure [here](@ref hdf5_input_structure)), which is given a suitable PEtab Id in the [YAML file](@ref YAML_file). This Id is then mapped to a PEtab variable in the mapping table. For example, given that `net1` has a file input, a valid mapping table would be:

| **petabEntityId** | **modelEntityId** |
|-------------------|-------------------|
| net1\_input\_file   | net1.input        |

Where `net1_input_file` has been specified in the problem [YAML file](@ref YAML_file).

When multiple simulation conditions each require a different neural network array input, the mapping table should map the input to a PEtab variable (for example, `net1_input`):

| **petabEntityId** | **modelEntityId** |
|-------------------|-------------------|
| net1\_input        | net1.input        |

This variable (here `net1_input`) should then assigned to specific input file variables (e.g. `net1_input_cond1` and `net1_input_cond2`) via the condition table. For a full example of a valid PEtab problem with array inputs, see [ADD].

TODO: Will fix an input format later.

### [Example: Network Observable Formula Output](@id output_obs)

If the neural network output appears in the observable formula, the PEtab entity should be directly referenced in the observable formula. For example, given:

| **petabEntityId** | **modelEntityId**  |
|-----------------|-----------------|
| net1\_output1    | net1.output[1]    |

A valid entry in the observable table would be:

| **observableId** | **observableFormula** |
|----------------|----------------------|
| obs1           | net1\_output1        |

As usual, the `observableFormula` can be any valid PEtab equation, so `net1_output1 + 1` would also be valid.

### Example: Network Scalar Output

If the output does not appear in the observable formula, the output variable should still be defined in the mapping table:

| **petabEntityId** | **modelEntityId**  |
|-----------------|-----------------|
| net1\_output1    | net1.output[1]    |

The output parameter (`net1_output1`) is then assigned in the condition or hybridization table.

## [Hybridization Table](@id hybrid_table)

The PEtab SciML extension introduces a new hybridization table for assigning neural network inputs and outputs. Assignments made in this table apply to all conditions, and together with the condition table, it specifies where a neural network is inserted in a PEtab SciML problem. The hybridization table has the following columns:

| **targetId**                | **operationType** | **targetValue**     |
|-------------------------|---------------|-----------------|
| NON\_ESTIMATED\_ENTITY\_ID | STRING        | MATH\_EXPRESSION |
| net1\_input1             | setValue      | p1              |
| net1\_input2             | setValue      | p1              |
| ...                     | ...           | ...             |

### Detailed Field Description

- `targetId` [NON\_ESTIMATED\_ENTITY\_ID, required]:
  The identifier of the non-estimated entity that will be modified. Restrictions vary depending on the `operationType` and the model type. Targets can be one of the following:
  - **Differential Targets**: Entities defined by a time derivative (e.g., targets of SBML rate rules or species that change by participating in reactions).
  - **Algebraic Targets**: Entities defined by an algebraic assignment (i.e., they are not associated with a time derivative and are generally not constant). In the context of a neural network, if the neural network appear in the observable formula or ODE RHS, its inputs are considered algebraic targets. If a neural network sets values prior to simulations, its inputs are considered to be a constant target.
  - **Constant Targets**: Entities defined by a constant value but may be subject to event assignments (e.g., SBML model parameters that are not targets of rate or assignment rules).
  - **Model Parameter Targets**: Entities corresponding to model parameters in the model file (e.g., SBML model parameters).
- `operationType` [STRING, required]:
  Specifies the type of operation to be performed on the target. Allowed values are:
  - `setValue`: Sets the current value of the target to the value specified in `targetValue`. The target must be a constant target.
  - `setRate`: Sets the time derivative of the target to `targetValue`. The target must be a differential target.
  - `setAssignment`: Sets the target to the symbolic value of `targetValue`. The target must be an algebraic target.
  - `setParameter`: Assigns the value of a model parameter to a neural network output. `targetValue` must be a neural network output.
- `targetValue` [STRING, required]:
  The value or expression that will be used to change the target. The interpretation of this value depends on the specified `operationType`.

### `operationType` and the Condition table Defines Hybrid Model Type

The `operationType` and condition table together specify where a neural network model is integrated into a PEtab SciML problem. In particular:

- **ODE RHS**: If all inputs use `setAssignment` and all outputs use either `setRate` or `setParameter`, the neural network appears in the ODE right-hand side.
- **Observable Formula**: If all inputs use `setAssignment` and no outputs are specified in the condition table, the neural network is part of the observable formula (note that the output variable must be referenced in the observable table).
- **Parameter/initial value setting**: If all inputs and outputs use `setValue`, or if they are only assigned in the condition table, the neural network sets model parameters or initial values before the simulation.

All other combinations are disallowed because they generally do not make sense in a PEtab context. For example, if inputs use `setAssignment` and outputs use `setValue`, then parameter values prior to simulation would be set by an assignment rule derived from model equations. This is not allowed in PEtab because assignment rules might be time-dependent. Furthermore, if a parameter is to be assigned via a rule, it should already be incorporated into the model. The `petab_sciml` library provides a linter to ensure that no disallowed combination is used. Packages that do not wish to depend on Python are strongly encouraged to verify that the input combinations in the condition table are valid.

!!! note "Model structure altering assignments"
    When `setRate`, `setAssignment`, or `setParameter` are used, the model structure or observable formula is altered during model import, effectively a neural network is inserted into the generated functions. Since such alterations apply to all conditions, these assignments can only be made in the hybridization table.

### [The Hybridization and the Condition table](@id hybrid_cond_tables)

When the neural network sets model parameters and/or initial values prior to simulations, assignments for **all conditions** can be set with the `setValue` operator type (see below). However, sometimes assignments need to be condition-specific. In such cases, the neural network inputs and/or outputs should be assigned in the condition table rather than in the hybridization table. For example, in the output case, if `net1_output1` sets `p1` in `cond1` and `p2` in `cond2`, a valid condition table is:

| **conditionId** | **targetId** | **targetValue** |
|-----------------|--------------|-----------------|
| cond1           | p1           | net1_output1    |
| cond2           | p2           | net1_output1    |

Similarly, for the input case, if `net1_input` is given by `p1` in `cond1` and `p2` in `cond2`, a valid condition table is:

| **conditionId** | **targetId** | **targetValue** |
|-----------------|--------------|-----------------|
| cond1           | net1_input   | p1              |
| cond2           | net1_input   | p2              |

!!! note "Assignments for a network variables occur either in hybridization or condition table"
    Since the hybridization table sets assignments for all conditions, variables assigned in the condition table cannot be assigned in the hybridization table, and vice versa. The linter will throw an error if a variable is assigned in both tables. If all assignments occur in the condition table, the hybridization table can be left empty or omitted.

### Examples: Assigning Neural Network Output

To set a **constant model parameter value** before model simulations for all conditions, the `setValue` operator type should be used. For example, if parameter `p` is determined by `net1_output1` (mapped to a neural network output in the mapping table) for all conditions, a valid hybridization table entry is:

| **operationType** | **targetId** | **targetValue** |
|------------------|--------------|-----------------|
| setValue         | p            | net1_output1    |

Similar, to set an **initial value**, the `setValue` operator type should be used. For example, if the initial value of species `X` is determined by `net1_output1` for all conditions, a valid hybridization table entry is:

| **operationType** | **targetId** | **targetValue** |
|------------------|--------------|-----------------|
| setInitial         | X            | net1_output1    |

These assignments above hold for all conditions. As explained [here](@ref hybrid_cond_tables), condition-specific assignments are also possible when the neural network sets variables prior to model simulations.

To set a **model derivative**, the `setRate` operator type should be used. For example, if the rate of species `X` is determined by `net1_output1`, a valid hybridization table entry is:

| **operationType** | **targetId** | **targetValue** |
|------------------|--------------|-----------------|
| setRate          | X            | net1_output1    |

To alter the **ODE RHS**, the `setParameter` operator type should be used.  For example, if an ODE model parameter `p` should be given by `net1_output1`, a valid hybridization table entry is:

| **operationType**    | **targetId** | **targetValue** |
|---------------------|--------------|-----------------|
| setParameter    | p            | net1\_output1    |

In theory for this last example, the `targetId` (here `p`) can also appear in the observable formula. However, as described [here](@ref output_obs), for the observable formula the mapped output variable must be directly encoded in the formula. Therefore, `targetId` appearing in the observable formula is not allowed, and the `petab_sciml` linter checks for this.

### Examples: Assigning Neural Network Input

When a neural network sets a **constant model parameter value or initial value**, its input variable (as specified in the mapping table) is treated as a standard PEtab variable. If that input variable is not defined in the parameter table, it should be assigned using the `setValue` operator type in the hybridization table if the assignment applies to all conditions; otherwise, it should be assigned in the condition table as explained [here](@ref hybrid_cond_tables).

When a neural network sets **a model derivative or alters the ODE RHS**, the input is considered to be an algebraic target. Therefore, the input variable should be assigned using the `setAssignment` operator. For example, if neural network input `net1_input1` is given by specie `X`, a valid condition table is:

| **operationType** | **targetId** | **targetValue** |
|------------------|-------------|---------------|
| setAssignment | net_input1  | X             |

## [Parameter Table](@id parameter_table)

The parameter table largely follows the same format as in PEtab version 2, with a subset of fields extended to accommodate neural network parameters. Further, a new `initializationPriorType` column is added for neural network-specific initialization. A general overview of the parameter table is available in the PEtab documentation; here, the focus is on extensions relevant to the SciML extension.

!!! note "Specific Assignments Have Precedence"
    When parsing, more specific assignments (e.g., `netId.layerId` rather than `netId`) take precedence for nominal values, priors, and so on. This means that if `netId` has one prior, and `netId.layerId` has another, the more specific assignment `netId.layerId` overrides the less specific one `netId` for `layerId` in this case.

### Detailed Field Description

- **parameterId [String, required]**: Identifies the neural network or a specific layer/parameter array. The target of the `parameterId` must be assigned via the [mapping table](@ref mapping_table).
- **nominalValue [String | NUMERIC, required]**: Specifies neural network nominal values. This can be:
  - A PEtab variable that via the problem [YAML file](@ref YAML_file) maps to an HDF5 file with the required [structure](@ref hdf5_ps_structure). If no file exists at the given path when the problem is imported and the parameters are set to be estimated, a file is created with randomly sampled values. Unless a numeric value is provided, referring to the same file is required for all assignments for a neural network, since all neural network parameters should be collected in a single HDF5 file following the structure described [here](@ref hdf5_ps_structure).
  - A numeric value applied to all parameters under `parameterId`.
- **estimate [0 | 1, required]**: Indicates whether the parameters are estimated (`1`) or fixed (`0`). This must be consistent across layers. For example, if `netId` has `estimate = 0`, then potential layer rows must also be `0`. In other words, freezing individual network parameters is not allowed.
- **initializationPriorType [String, optional]**: Specifies the prior used for sampling initial values before parameter estimation. In addition to the PEtab-supported priors [ADD], the SciML extension supports the following standard neural network initialization priors:
  - [`kaimingUniform`](https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_uniform_) (default) — with `gain` as `initializationPriorParameters` value.
  - [`kaimingNormal`](https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_normal_) — with `gain` as `initializationPriorParameters` value.
  - [`xavierUniform`](https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.xavier_uniform_) — with `gain` as `initializationPriorParameters` value.
  - [`xavierNormal`](https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.xavier_normal_) — with `gain` as `initializationPriorParameters` value.
- - **initializationPriorParameters [NUMERIC, optional]**: Distribution parameter value for the `initializationPriorType`. Which parameter(s) are referred to depends on the chosen prior distribution.

### Bounds for neural net parameters

Bounds can be specified for an entire network or its nested levels. However, it should be noted that most optimization algorithms used for neural networks, such as ADAM, do not support parameter bounds in their standard implementation.

### Example: Different Priors for Different Layers

Different layers can have distinct initialization prior parameters. For example, consider a neural-network model `net1` where `layer1` and `layer2` require different `initializationPriorParameters` because they use different activation functions that need distinct [gain](https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.calculate_gain) values for the `kaimingUniform` prior. A valid parameter table would be:

| **parameterId**    | **parameterScale** | **lowerBound** | **upperBound** | **estimate** | **nominalValue** | **initializationPriorType** | **initializationPriorParameters** |
|--------------------|--------------------|----------------|----------------|------------|------------------|-----------------------------|-------------------------------------|
| net1\_ps            | lin                | -inf           | inf            | 1          | net1\_ps\_file      | kaimingUniform              | 1                                   |
| net1\_layer1_ps     | lin                | -inf           | inf            | 1          | net1\_ps\_file      | kaimingUniform              | 1                                   |
| net1\_layer2_ps     | lin                | -inf           | inf            | 1          | net1\_ps\_file      | kaimingUniform              | 5/3                                 |

Where `parameterId` are assumed to have been properly assigned in the [mapping table](@ref mapping_table). In this example, each layer references the same file variable for `nominalValue`. This means the layers obtain their values from the specified file.

It is also possible to specify different priors for different parameter arrays within a layer. For example, to use different priors for the weights and bias in `layer1` of `net1` (assuming a `linear` layer), a valid parameter table would be:

| **parameterId**      | **parameterScale** | **lowerBound** | **upperBound** | **estimate** | **nominalValue** | **initializationPriorType** | **initializationPriorParameters** |
|----------------------|--------------------|----------------|----------------|------------|------------------|-----------------------------|-------------------------------------|
| net1\_ps              | lin                | -inf           | inf            | 1          | net1\_ps\_file      | kaimingUniform              | 1                                   |
| net1\_layer1_weight   | lin                | -inf           | inf            | 1          | net1\_ps\_file      | kaimingUniform              | 1                                   |
| net1\_layer1_bias     | lin                | -inf           | inf            | 1          | net1\_ps\_file      | kaimingNormal               | 5/3                                 |

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
