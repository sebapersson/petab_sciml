# Format Specification

A PEtab SciML problem extends the PEtab standard version 2 to accommodate hybrid models (SciML problems) that combine neural network (NN) and mechanistic components. Two new file types are introduced by the extension:

1. [Neural Network File(s)](@ref nn_format): Optional YAML file(s) describing NN model(s).
2. [Hybridization table](@ref hybrid_table): Table for assigning NN outputs and inputs.

PEtab SciML further extends the following standard PEtab files:

1. [Mapping Table](@ref mapping_table): Extended to describe how NN inputs, outputs and parameters map to PEtab entities.
2. [Parameters Table](@ref parameter_table): Extended to describe nominal values for NN parameters.
3. [Problem YAML File](@ref YAML_file): Extended to include a new SciML field for NN models and (optionally) array or tensor formatted data.

All other PEtab files remain unchanged. This specification explains the format for each file that is added or modified by the PEtab SciML extension.

## [High Level Overview](@id hybrid_types)

The PEtab SciML specification is designed to keep the mechanistic model, NN model, and PEtab problem as independent as possible while linking them through the hybridization and/or condition tables. In this context, mechanistic models are typically defined using community standards like SBML and are commonly simulated as systems of ordinary differential equations (ODEs), and here the terms mechanistic model and ODE are used interchangeably. Essentially, the PEtab SciML approach takes a PEtab problem involving a mechanistic ODE model and supports the integration of NN inputs and outputs.

PEtab SciML supports two classes of hybrid models:

1. **Static hybridization**: For each experimental/simulation condition, inputs are constant and the NN model sets constant parameters and/or initial values in the ODE model prior to model simulation.
2. **Dynamic hybridization**: The NN model appears in the ODE right-hand-side (RHS) and/or observable formula. Inputs and outputs are computed dynamically over the course of a simulation.

A PEtab SciML problem can also include multiple NNs. Aside from ensuring that NNs do not conflict (e.g., by sharing the same output), no special considerations are required. Each additional NN is included just as it would be in the single NN case.

## [NN Model YAML Format](@id nn_format)

The NN model format is flexible, meaning models can be provided in any format compatible with the PEtab SciML specification (for example, [Lux.jl](https://github.com/LuxDL/Lux.jl) in [PEtab.jl](https://github.com/sebapersson/PEtab.jl)). Additionally, the `petab_sciml` library provides a NN model YAML format that can be imported by tools across various programming languages.

!!! tip "For everyone: Use the NN model YAML format for interoperability"
    The NN model specification format in PEtab SciML is flexible, to ensure all architectures can be used. However, where possible, the NN model YAML format should be used, to facilitate model exchange.

A NN model must consist of two parts to be compatible with the PEtab SciML specification:

- **layers**: Defines the NN layers, each with a unique identifier.
- **forward**: A forward pass function that, given input arguments, specifies the order in which layers are called, applies any activation functions, and returns one or several arrays. The forward function can accept more than one input argument (`n > 1`), and in the [mapping table](@ref mapping_table), the forward function's `n`th input argument (ignoring any potential class arguments such as `self`) is referred to as `inputArgumentIndex{n-1}`. Similar holds for the output. Aside from the NN output values, every component that should be visible to other parts of the PEtab SciML problem must be defined elsewhere (e.g., in **layers**).

## [Array data](@id hdf5_array)

The standard PEtab format is unsuitable for incorporating large arrays of values into an estimation problem. This includes the large datasets used to train NNs, or the parameter values of deep NNs.

Hence, we provide a HDF5-based file format to store and incorporate this array data efficiently. The format involves a list of arrays, each of which can optionally be used for a specific subset of conditions. The arrays can optionally be assigned to a specific target within the HDF5 file, or in the NN model YAML file. There is an additional metadata section to specify the array order (i.e., row-major or column-major arrays).

The general structure is
```hdf5
my_array.hdf5                        # arbitrary filename
├ metadata
│ └── perm                           # reserved keyword (str). "row" for row-major, "column" for column-major
└ arrays
  ├─┬─ conditionIds                  # reserved keyword. (optional) The list of applicable conditions.
  │ ├─ targetId                      # reserved keyword. (optional) The model component that will be assigned this array value.
  │ └─ targetValue                   # reserved keyword. The tensor.
  └─ ...
```

The schema is provided as [JSON schema](assets/arrays_schema.json). Currently, validation is only provided via the PEtab SciML library.

There are two ways to specify that the arrays are input data:

- set the `targetId` to an input Id. This could be an input Id from the NN model YAML file, or an Id in the `$nnId.inputs{[$inputArgumentIndex]{[$inputIndex]}}` [format](@ref nn_inputs)
- omit `targetId` and associate the arrays of a file to a model target elsewhere in the PEtab SciML problem. This can be in the NN model YAML, or the (TODO GH issue #17) mapping table

There are three ways to specify that the arrays are layer parameter values:

- set the `targetId` to a fully-qualified parameter name, in the `$nnId.parameters[$layerId].{[$arrayId]{[$parameterIndex]}}` [format](@ref nn_parameters)
- set the `targetId` to a layer-specific parameter name, such as "weight" for a PyTorch Linear layer. This then needs to be associated to a Linear layer in either in NN model YAML, or the (TODO GH issue #17) mapping table
- omit `targetId` and associate the arrays of a file to a parameter elsewhere in the PEtab SciML problem. This can be in the NN model YAML `layers` section, or the (TODO GH issue #17) mapping table

The naming of inputs or layer parameters is framework-specific or user-specified.
For inputs:

- NN models in the PEtab SciML [YAML format](@ref NN_YAML) follow PyTorch indexing. For example, if the first layer is `Conv2d`, the input should be in `(C, W, H)` format.
- NN models in other formats follow the indexing and naming conventions of the respective package and programming language.

For parameters:

- NN models in the PEtab SciML [YAML format](@ref NN_YAML) follow PyTorch indexing and naming conventions. For example, in a PyTorch `Linear` layer, the arrays Ids are `weight` and/or `bias`
- NN models in other formats follow the indexing and naming conventions of the respective package and programming language.

!!! tip "For developers: Respect memory order"
    Tools supporting the SciML extension should, for computational efficiency, reorder input data and potential layer parameter arrays to match the memory ordering of the target language. For example, PEtab.jl converts input data to follow Julia based indexing.

!!! tip "For developers: Allow export of parameters in PEtab SciML format"
    If the NN is not provided in the YAML format, exchange of NN parameters between software is not possible. To facilitate exchange, it is recommended that tools supporting PEtab SciML implement a function capable of exporting to the PEtab SciML format if all layers in the NN correspond to layers supported by the PEtab SciML NN model YAML format.

### [NN model YAML format](@id NN_YAML)

The `petab_sciml` library provides a NN model YAML format for model exchange. This format follows PyTorch conventions for layer names and arguments. The schema is provided as [JSON schema](assets/nn_model_schema.json), which enables validation with various third-party tools.

!!! tip "For users: Define models in PyTorch"
    The recommended approach to create a NN model YAML file is to first define a PyTorch model (`torch.nn.Module`) and use the Python `petab_sciml` library to export this to YAML. See the tutorials for examples of this.

## [Mapping Table](@id mapping_table)

All NNs are assigned an identifier in the PEtab problem [YAML](@ref YAML_file) file. A NN identifier is not considered a valid PEtab identifier, to avoid confusion about what it refers to (e.g., parameters, inputs, outputs). Consequently, every NN input, parameter, and output referenced in the PEtab problem must be defined under `modelEntityId` and mapped to a PEtab identifier. For the `PEtabEntityId` column the same rules as in PEtab v2 apply. Additionally array file Ids defined in the [YAML](@ref YAML_file) file are considered valid PEtab entities.

### `modelEntityId` [STRING, REQUIRED]

A modeling-language-independent syntax which refers to inputs, outputs, and parameters of NNs.

#### [Parameters](@id nn_parameters)

The model Id `$nnId.parameters[$layerId].{[$arrayId]{[$parameterIndex]}}` refers to the parameters of a NN identified by `$nnId`.

- `$layerId`: The unique identifier of the layer (e.g., `conv1`).
- `$arrayId`: The parameter array name specific to that layer (e.g., `weight`).
- `$parameterIndex`: The indexing into the parameter array ([syntax](@ref mapping_table_indexing)).

NN parameter PEtab identifiers can only be referenced in the parameters table.

#### [Inputs](@id nn_inputs)

The model Id `$nnId.inputs{[$inputArgumentIndex]{[$inputIndex]}}` refers to specific inputs of the NN identified by `$nnId`.

- `$inputArgumentIndex`: The input argument number in the NN forward function. Starts from 0.
- `$inputIndex` Indexing into the input argument ([syntax](@ref mapping_table_indexing)). Should not be specified if the input is a file.

For [static hybridization](@ref hybrid_types) NN input PEtab identifiers are considered valid PETAB\_IDs without restrictions (e.g., they may be referenced in the parameters table, condition table, hybridization table, etc.). For [dynamic hybridization](@ref hybrid_types), input PEtab identifiers can only be assigned an expression in the [hybridization table](@ref hybrid_table).

#### Outputs

The model Id `$nnId.outputs{[outputArgumentIndex]{[$outputIndex]}}` refers to specific outputs of a NN identified by `$nnId`.

- `$outputId`: The output argument number in the NN forward function. Starts from 0.
- `$outputIndex`: Indexing into the output argument ([syntax](@ref mapping_table_indexing))

#### Nested Identifiers

The PEtab SciML extension supports nested identifiers for mapping structured or hierarchical elements. Identifiers are expressed in the hierarchical indicated above using nested curly brackets. Valid examples are:

- `nn1.parameters`
- `nn1.parameters[conv1]`
- `nn1.parameters[conv1].weight`

!!! warn "Do not break the hierarchy"
    Identifiers that break the hierarchy (e.g., `nn1.parameters.weight`) are not valid.

#### [Indexing](@id mapping_table_indexing)

Indexing into arrays follows the format `[i0, i1, ...]`, and indexing notation depends on the NN library:

- NN models in the PEtab SciML [NN model YAML format](@ref NN_YAML) follow PyTorch indexing. Consequently, indexing is 0-based.
- NN models in other formats follow the indexing and naming conventions of the respective package and programming language.

#### Assigning Values

For assignments to nested PEtab identifiers (in the `parameters`, `hybridization`, or `conditions` tables), assigned values must either:

- Refer to another PEtab identifier with the same nested structure, or
- Follow the corresponding hierarchical HDF5 [input](@ref hdf5_input_structure) or [parameter](@ref hdf5_ps_structure) structure.

## [Hybridization Table](@id hybrid_table)

A tab-separated values file for assigning NN inputs and outputs. Assignments in the table the table apply to all simulation conditions. Expected to have, in any order, the following two columns:

| **targetId**                | **targetValue**     |
|-------------------------|---------------|
| NON\_ESTIMATED\_ENTITY\_ID | MATH\_EXPRESSION |
| nn1\_input1             | p1              |
| nn1\_input2             | p1              |
| ...                     | ...             |

### Detailed Field Description

- `targetId` [NON\_ESTIMATED\_ENTITY\_ID, REQUIRED]: The identifier of the non-estimated entity that will be modified. Restrictions depend on hybridization type ([static- or dynamic hybridization](@ref hybrid_types)). See below.
- `targetValue` [STRING, REQUIRED]: The value or expression that will be used to change the target.

### Static hybridization

Static hybridization NN model inputs and outputs are constant targets (case 1 [here](@ref hybrid_types)).

#### Inputs

Valid `targetValue`'s for a NN input are:

- A parameter in the parameter table.
- An array input file (assigned an Id in the [YAML problem file](@ref YAML_file)).

#### Outputs

Valid `targetId`'s for a NN output are:

- A non-estimated model parameter.
- A species' initial value (referenced by the species' Id). In this case, any other species initialization is overridden.

#### Condition and Hybridization Tables

NN input variables are valid `targetId`s for the condition table as long as, following the PEtab standard, they are NON\_PARAMETER\_TABLE\_ID. **Importantly**, since the hybridization table defines assignments for all simulation conditions, any `targetId` value in the condition table cannot appear in the hybridization table, and vice versa.

NN output variables can also appear in the `targetValue` column of the condition table.

### Dynamic hybridization

Dynamic hybridization NN models depend on model simulated model quantities (case 2 [here](@ref hybrid_types)).

#### Inputs

Valid `targetValue` for a NN input is an expression that depend on model species, time, and/or parameters. Any model species and/or parameters in the expression are expected to be evaluated at the given time-value.

#### Outputs

Valid `targetId` for a NN output is a constant model parameter. During PEtab problem import, any assigned parameters is replaced by the NN output in the ODE RHS.

## [Parameter Table](@id parameter_table)

The parameter table follows the same format as in PEtab version 2, with a subset of fields extended to accommodate NN parameters. This section focuses on columns extended by the SciML extension.

!!! note "Specific Assignments Have Precedence"
    More specific assignments (e.g., `nnId.parameters[layerId]` instead of `nnId.parameters`) have precedence for nominal values, priors, and other setting. For example, if a nominal values is assigned to `nnId.parameters` and a different nominal value is assigned to `nnId.parameters[layerId]`, the latter is used.

### Detailed Field Description

- `parameterId` [String, REQUIRED]: The NN or a specific layer/parameter array id. The target of the `parameterId` must be assigned via the [mapping table](@ref mapping_table).
- `nominalValue` [String | NUMERIC, REQUIRED]: NN nominal values. This can be:
  - A PEtab variable that via the problem [YAML file](@ref YAML_file) corresponds to an HDF5 file with the required [structure](@ref hdf5_ps_structure). If no file exists at the given path when the problem is imported and the parameters are set to be estimated, a file is created with randomly sampled values. Unless a numeric value is provided, referring to the same file is required for all assignments for a NN, since all NN parameters should be collected in a single HDF5 file following the structure described [here](@ref hdf5_ps_structure).
  - A numeric value applied to all parameters under `parameterId`.
- `estimate` [0 | 1, REQUIRED]: Indicates whether the parameters are estimated (`1`) or fixed (`0`).

### Bounds for NN parameters

Bounds can be specified for an entire NN or its nested identifiers. However, most optimization algorithms used for NNs, such as ADAM, do not support parameter bounds in their standard implementations. Therefore, NN bounds are optional and default to `-inf` for the lower bound and `inf` for the upper bound.

## [Problem YAML File](@id YAML_file)

PEtab SciML files are defined within the `extensions` section of a PEtab YAML file. This section specifies the configuration of NNs and optional array files used for simulation or parameter estimation.

### Fields

#### `neural_networks` [REQUIRED]

A list of NN definitions. Each entry is a mapping with the following keys:

- `location` [STRING]: File path to the NN model.
- `format` [STRING]: Format of the NN. Use `YAML` if the model is defined in the [PEtab SciML YAML format](@ref NN_YAML). For models defined using external libraries, specify the library name (e.g., `Lux.jl`, `equinox.py`).
-`dynamic` [BOOL]: Indicates the hybridization type (see [Hybrid Types](@ref hybrid_types)):
  - `true`: dynamic hybridization
  - `false`: static hybridization

#### `array_files` [OPTIONAL]

A list of array file definitions. Each entry is a mapping with the following keys:

- **`location`** [STRING]: File path to the array file.
- **`format`** [STRING]: Format of the file (e.g., `HDF5`).

Parameter array files must follow the structure described in [HDF5 Parameter Structure](@ref hdf5_ps_structure). Input array files must follow the structure described in [HDF5 Input Structure](@ref hdf5_input_structure).

If a NN is provided in another format than the YAML format, respective tool must provide the NN during problem import. Note that regardless of NN model format, for exchange purposes the NN model **must** be available in a file (not in the main PEtab problem import script).
