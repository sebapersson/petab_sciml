# Format Specification

A PEtab SciML problem extends the PEtab standard version 2 to accommodate hybrid models (SciML problems) that combine neural network and mechanistic components. Two new file types are introduced by the extension:

1. [Neural Net File(s)](@ref net_format): Optional YAML file(s) describing neural net model(s).
2. [Hybridization table](@ref hybrid_table): Table assigning neural network outputs and inputs.

The extension further extends the following standard PEtab files:

1. [Mapping Table](@ref mapping_table): Extended to describe how neural network inputs and outputs map to PEtab variables.
2. [Parameters Table](@ref parameter_table): Extended to describe nominal values for network parameters.
3. [Problem YAML File](@ref YAML_file): Extended to include a new SciML field for neural network models and (optionally) array or tensor formatted data.

All other PEtab files remain unchanged. This specification explains the format for each file that is added or modified by the PEtab SciML extension.

## [High Level Overview](@id hybrid_types)

The PEtab SciML specification is designed to keep the dynamic model, neural network model, and PEtab problem as independent as possible while linking them through the hybridization and/or condition tables. In this context, mechanistic models are typically defined using community standards like SBML and are commonly simulated as systems of ordinary differential equations (ODEs), and here the terms mechanistic model and ODE are used interchangeably. Essentially, the PEtab SciML approach takes a PEtab problem involving a mechanistic ODE model and supports the integration of neural network inputs and outputs.

PEtab SciML supports two types of hybridizations (or classes of hybrid models):

1. **Pre simulation hybridization**: The neural network model sets constant parameters and/or initial values in the ODE model prior to model simulation. Inputs are constant per simulation condition.
2. **Intra simulation hybridization**: The neural network model appears in the ODE RHS and/or observable formula. Inputs are per time-point computed from simulated quantities.

A PEtab SciML problem can include multiple neural networks. Aside from ensuring that neural networks do not conflict (e.g., by sharing the same target), no special considerations are required. Each additional network is included just as it would be in the single-network case.

## [Neural Network Model Format](@id net_format)

The neural network model format is flexible, and models can be provided in any format supported by tools compatible with the PEtab SciML format (for instance, [Lux.jl](https://github.com/LuxDL/Lux.jl) in [PEtab.jl](https://github.com/sebapersson/PEtab.jl)). Additionally, the `petab_sciml` library offers a neural network YAML file format that can be imported by tools across various programming languages. his format flexibility exists because, although the YAML format can accommodate many architectures, some may still be difficult to represent. However, when possible, the YAML format is recommended to facilitate model exchange across different software.

A neural network model must consist of two parts to be compatible with the PEtab SciML problem files:

- **layers**: A constructor that defines the network layers, each with a unique identifier.
- **forward**: A forward pass function that, given input arguments, specifies the order in which layers are called, applies any activation functions, and returns one or several arrays. The forward function can accept more than one input argument (`n > 1`), and in the [mapping table](@ref mapping_table), the forward function's `n`th input argument (ignoring any potential class arguments such as `self`) is referred to as `inputArgumentIndex{n}`. Similar holds for the output. Aside from the neural network output values, every component that should be visible to other parts of the PEtab SciML problem must be defined elsewhere (e.g., in **layers**).

### [Neural Network Parameters](@id hdf5_ps_structure)

All parameters for a neural network model are stored in an HDF5 file, with the file path specified in the problem [YAML file](@ref YAML_file). The HDF5 parameter file is expected to have the following structure for an arbitrary number of layers:

```hdf5
parameters.hdf5
└───layerId{1} (group)
│   ├── arrayId{1}
│   └── arrayId{2}
└───layerId{2} (group)
    ├── arrayId{1}
    └── arrayId{2}
```

The indexing convention and naming for `arrayId` are determined by the neural network model library:

- Neural network models in the PEtab SciML [YAML format](@ref YAML_net_format) follow PyTorch indexing and naming conventions. For example, in a PyTorch `linear` layer, the arrays are identified as `weight` and (optionally) `bias`
- Neural network models in other formats follow the indexing and naming conventions of the respective package and programming language.

### [Neural Network Input](@id hdf5_input_structure)

Potential neural network input files are expected to have the following format:

```hdf5
input.hdf5
└───input (group)
    └─── input_array
```

As with [parameters](@ref hdf5_ps_structure), the indexing depends on the neural network library:

- Neural network models in the PEtab SciML [YAML format](@ref YAML_net_format) follow PyTorch indexing. For example, if the first layer is `Conv2d`, the input should be in `(C, W, H)` format.
- Neural network models in other formats follow the indexing and naming conventions of the respective package and programming language.

!!! tip "For developers: Respect memory order"
    Tools supporting the SciML extension should, for computational efficiency, reorder input data and potential layer parameter arrays to match the memory ordering of the target language. For example, PEtab.jl converts input data to follow Julia based indexing.

TODO: We will fix condition specific input in the YAML file later.

### [YAML Network file format](@id YAML_net_format)

The `petab_sciml` library provides a YAML neural network file format for model exchange. The YAML format follows PyTorch conventions for layer names and arguments. YAML files can be written manually, however, it is recommended approach to define a PyTorch `nn.Module` and use the `petab_sciml` library to automatically generate the YAML representation (see tutorials).

TODO: Maybe we should provide some scheme of the YAML here, Dilan?

## [Mapping Table](@id mapping_table)

All neural networks are assigned an Id in the PEtab problem [YAML](@ref YAML_file) file. A neural network Id is not considered a valid PEtab identifier, to prevent confusion regarding what its refers to (e.g., parameters, inputs, outputs). Consequently, every neural network input, parameter, and output referenced in the PEtab problem must be defined in the mapping table under `modelEntityId` and mapped to a PEtab identifier. Note that array file IDs defined in the [YAML](@ref YAML_file) file are considered valid entities in the `PEtabEntityId` column.

### `modelEntityId` [STRING, REQUIRED]

A modeling-language-independent syntax which refers to inputs, outputs, and parameters of neural networks.

#### Parameters

The model Id `$netId.parameters[$layerId].{$arrayId}[$parameterIndex]` is used to reference individual parameters of a neural network identified by `$netId`.

- `$layerId`: The unique identifier of the layer (e.g., `conv1`).
- `$arrayId`: The parameter array name specific to that layer (e.g., `weight`).
- `$parameterIndex`: The indexing into the parameter array ([syntax](@ref mapping_table_indexing)).

#### Inputs

The model ID `$netId.inputs[$inputArgumentIndex][$inputIndex]` refers to specific inputs of the network identified by `$netId`.

- `$inputArgumentIndex`: The input argument number in the neural network forward function. Starts from 1.
- `$inputIndex` Indexing into the input argument ([syntax](@ref mapping_table_indexing)). Should not be specified if the input is a file.

#### Outputs

The model ID `$netId.outputs[outputArgumentIndex][$outputIndex]` refers to specific outputs of a neural network identified by `$netId`.

- `$outputId`: The output argument number in the neural network forward function. Starts from 1.
- `$outputIndex`: Indexing into the output argument ([syntax](@ref mapping_table_indexing))

#### Nested Identifiers

The PEtab SciML extension supports **nested identifiers** for mapping structured or hierarchical elements. Identifiers are expressed in a hierarchical format using nested curly brackets. Valid examples are:

- `net1.parameters`
- `net1.parameters[conv1]`
- `net1.parameters[conv1].weight`

!!! warn "Do not break the hierarchy"
    Identifiers that break the hierarchy (e.g., `net1.parameters.weight`) are not valid.

#### [Indexing](@id mapping_table_indexing)

Indexing into arrays depends on the neural network library:

- Neural network models in the PEtab SciML [YAML format](@ref YAML_net_format) follow PyTorch indexing. Consequently, indexing is 0-based.
- Neural network models in other formats follow the indexing and naming conventions of the respective package and programming language.

#### Assigning Values

For assignments to nested PEtab identifiers (in the `parameters`, `hybridisations`, or `conditions` tables), assigned values must either:

- Refer to another PEtab identifier with the same nested structure, or
- Follow the corresponding hierarchical HDF5 [input](@ref hdf5_input_structure) or [parameter](@ref hdf5_ps_structure) structure.

## [Hybridization Table](@id hybrid_table)

A tab-separated values file which assigns neural network inputs and outputs. Assignments in the table the table apply to all simulation conditions. Expected to have, in any order, the following two columns:

| **targetId**                | **targetValue**     |
|-------------------------|---------------|
| NON\_ESTIMATED\_ENTITY\_ID | MATH\_EXPRESSION |
| net1\_input1             | p1              |
| net1\_input2             | p1              |
| ...                     | ...             |

### Detailed Field Description

- `targetId` [NON\_ESTIMATED\_ENTITY\_ID, REQUIRED]: The identifier of the non-estimated entity that will be modified. Restrictions depend on hybridization type ([pre- or intra-simulation hybridization](@ref hybrid_types)). See below.
- `targetValue` [STRING, REQUIRED]: The value or expression that will be used to change the target.

### Pre-simulation hybridization

Pre-simulation neural network model inputs and outputs are constant targets (case 2 [here](@ref hybrid_types)).

#### Inputs

Valid `targetValue`'s for a neural network input are:

- A parameter in the parameter table
- An array input file (assigned an Id in the [YAML problem file](@ref YAML_file)).

#### Outputs

Valid `targetId`'s for a neural network output are:

- A non-estimated model parameter
- A specie's initial value (referenced by the specie's Id). In this case, any other specie initialization is overridden.

#### Condition and Hybridization Tables

Pre-simulation assignments can also be made in the conditions table, and combinations are permitted. For example, all inputs can be assigned in the condition table while all outputs are assigned in the hybridization table. However, since the hybridization table defines assignments for all simulation conditions, any `targetId` assigned in the condition table cannot be assigned in the hybridization table, and vice versa.

### Intra-simulation hybridization

Intra-simulation neural network models depend on model simulated model quantities (case 2 [here](@ref hybrid_types)).

#### Inputs

Valid `targetValue` for an input is an expression that depend on model species, time, and/or parameters. Any species and/or parameters in the expression are expected to be evaluated at the given time-value.

#### Outputs

Valid `targetId` for an output is a constant model parameter. During PEtab problem import, assigned parameters are replaced by the neural network output in the ODE RHS.

An important note is that it is considered invalid for a model parameter assigned by a neural network output to appear in both the ODE RHS and the observable formula. In the observable formula, neural network output PEtab identifiers are expected to be directly encoded, making a neural network–assigned parameter in the formula ambiguous. Conversely, it is considered valid if a neural network output is used to assign an ODE model parameter and is also directly encoded in the observable formula.

## [Parameter Table](@id parameter_table)

The parameter table follows the same format as in PEtab version 2, with a subset of fields extended to accommodate neural network parameters. This section focuses on columns extended by the SciML extension.

!!! note "Specific Assignments Have Precedence"
    More specific assignments (e.g., `netId.layerId` instead of `netId`) have precedence for nominal values, priors, and other setting. For example, if a nominal values is assigned to `netId` and a different nominal value is assigned to `netId.layerId`, the latter is used in place of the former for `netId.layerId`.

### Detailed Field Description

- **parameterId [String, REQUIRED]**: The neural network or a specific layer/parameter array id. The target of the `parameterId` must be assigned via the [mapping table](@ref mapping_table).
- **nominalValue [String | NUMERIC, REQUIRED]**: Neural network nominal values. This can be:
  - A PEtab variable that via the problem [YAML file](@ref YAML_file) maps to an HDF5 file with the required [structure](@ref hdf5_ps_structure). If no file exists at the given path when the problem is imported and the parameters are set to be estimated, a file is created with randomly sampled values. Unless a numeric value is provided, referring to the same file is required for all assignments for a neural network, since all neural network parameters should be collected in a single HDF5 file following the structure described [here](@ref hdf5_ps_structure).
  - A numeric value applied to all parameters under `parameterId`.
- **estimate [0 | 1, REQUIRED]**: Indicates whether the parameters are estimated (`1`) or fixed (`0`). This must be consistent across layers. For example, if `netId` has `estimate = 0`, then potential layer rows must also be `0`. In other words, freezing individual network parameters is not allowed.

### Bounds for neural net parameters

Bounds can be specified for an entire network or its nested identifiers. However, most optimization algorithms used for neural networks, such as ADAM, do not support parameter bounds in their standard implementations. Therefore, neural net bounds are optional and default to `-inf` for the lower bound and `inf` for the upper bound.

## [Problem YAML File](@id YAML_file)

An extension section is included in the PEtab SciML YAML file for specifying neural network YAML files, as well as array parameter, input, and output files. These elements are defined using the following key-value mappings:

- `file[extensions][petab_sciml][neural_nets]`: Neural network models. Here, each network is defined as a key-value mapping, where the key is the unique neural network ID (`netId`), and the corresponding value is another key-value mapping:
  - `[netId][location]`: The file path where the neural network model is stored.
  - `[netId][format]`: The neural network format. Expected to be `YAML` if the network is provided in the PEtab SciML library [YAML format](@ref YAML_net_format). Otherwise, the neural network library should be provided (e.g Lux.jl or equinox.py).
  - `[netId][hybridization]`: The neural network hybridization type. Expected to be either `pre_simulation` or `intra_simulation` (for type information see [here](@ref hybrid_types)).
- `yaml_file[extensions][petab_sciml][array_files]`  Potential array files. Parameter files are expected to follow the structure described [here](@ref hdf5_ps_structure), and input files should follow the structure described [here](@ref hdf5_input_structure). Each entry is defined with a key-value mapping where the key is the array ID (`arrayId`), and the corresponding value is another key-value mapping:
  - `[arrayId][location]`: The file path.
  - `[arrayId][format]`: The file format (e.g., HDF5).

If a neural network is provided in another format than the YAML format, respective tool must provide the network during problem import. Note that regardless of neural-network format, for exchange purposes the neural network model **must** be available in a file.
