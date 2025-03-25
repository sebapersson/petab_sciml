var documenterSearchIndex = {"docs":
[{"location":"net_activation.html#layers_activation","page":"Supported Layers and Activation Functions","title":"Supported Layers and Activation Functions","text":"","category":"section"},{"location":"net_activation.html","page":"Supported Layers and Activation Functions","title":"Supported Layers and Activation Functions","text":"The PEtab SciML neural network model YAML format supports numerous standard neural network layers and activation functions. Layer names and associated keyword arguments follow the PyTorch naming scheme. PyTorch is used because it is currently the most popular machine learning framework, and its comprehensive documentation makes it easy to look up details for any specific layer or activation function.","category":"page"},{"location":"net_activation.html","page":"Supported Layers and Activation Functions","title":"Supported Layers and Activation Functions","text":"If support is lacking for a layer or activation function you would like to see, please file an issue on GitHub.","category":"page"},{"location":"net_activation.html","page":"Supported Layers and Activation Functions","title":"Supported Layers and Activation Functions","text":"The table below lists the supported and tested neural network layers along with links to their respective PyTorch documentation. Additionally, the table indicates which tools support each layer.","category":"page"},{"location":"net_activation.html","page":"Supported Layers and Activation Functions","title":"Supported Layers and Activation Functions","text":"layer PEtab.jl AMICI\nLinear ✔️ \nBilinear ✔️ \nFlatten ✔️ \nDropout ✔️ \nDropout1d ✔️ \nDropout2d ✔️ \nDropout3d ✔️ \nAlphaDropout ✔️ \nConv1d ✔️ \nConv2d ✔️ \nConv3d ✔️ \nConvTranspose1d ✔️ \nConvTranspose2d ✔️ \nConvTranspose3d ✔️ \nMaxPool1d ✔️ \nMaxPool2d ✔️ \nMaxPool3d ✔️ \nAvgPool1d ✔️ \nAvgPool2d ✔️ \nAvgPool3d ✔️ \nLPPool1 ✔️ \nLPPool2 ✔️ \nLPPool3 ✔️ \nAdaptiveMaxPool1d ✔️ \nAdaptiveMaxPool2d ✔️ \nAdaptiveMaxPool3d ✔️ \nAdaptiveAvgPool1d ✔️ \nAdaptiveAvgPool2d ✔️ \nAdaptiveAvgPool3d ✔️ ","category":"page"},{"location":"net_activation.html#Supported-Activation-Function","page":"Supported Layers and Activation Functions","title":"Supported Activation Function","text":"","category":"section"},{"location":"net_activation.html","page":"Supported Layers and Activation Functions","title":"Supported Layers and Activation Functions","text":"The table below lists the supported and tested activation functions along with links to their respective PyTorch documentation. Additionally, the table indicates which tools support each layer.","category":"page"},{"location":"net_activation.html","page":"Supported Layers and Activation Functions","title":"Supported Layers and Activation Functions","text":"Function PEtab.jl AMICI\nrelu ✔️ \nrelu6 ✔️ \nhardtanh ✔️ \nhardswish ✔️ \nselu ✔️ \nleaky_relu ✔️ \ngelu ✔️ \ntanhshrink ✔️ \nsoftsign ✔️ \nsoftplus ✔️ \ntanh ✔️ \nsigmoid ✔️ \nhardsigmoid ✔️ \nmish ✔️ \nelu ✔️ \ncelu ✔️ \nsoftmax ✔️ \nlog_softmax ✔️ ","category":"page"},{"location":"test_info.html#Test-Suite","page":"Test Suite","title":"Test Suite","text":"","category":"section"},{"location":"test_info.html","page":"Test Suite","title":"Test Suite","text":"The PEtab SciML format provides an extensive test suite to verify the correctness of tools that support the standard. The tests are divided into three parts, which are recommended to be run in this order:","category":"page"},{"location":"test_info.html","page":"Test Suite","title":"Test Suite","text":"Neural network import\nInitialization (start guesses for parameter estimation)\nHybrid models that combine mechanistic and data-driven components","category":"page"},{"location":"test_info.html","page":"Test Suite","title":"Test Suite","text":"The neural network tests cover a relatively large set of architectures. The hybrid tests, by comparison, involve fewer network architectures, because if the hybrid interface works with a given library (e.g., Equinox or Lux.jl) and the implementation cleanly separates neural network and dynamic components, the combination should function correctly once network models can be imported properly.","category":"page"},{"location":"test_info.html#Neural-network-import-tests","page":"Test Suite","title":"Neural-network import tests","text":"","category":"section"},{"location":"test_info.html","page":"Test Suite","title":"Test Suite","text":"The neural networks test different layers and activation functions. A complete list of tested layers and activation functions can be found here.","category":"page"},{"location":"test_info.html#Initialization-tests","page":"Test Suite","title":"Initialization tests","text":"","category":"section"},{"location":"test_info.html","page":"Test Suite","title":"Test Suite","text":"These tests ensure that nominal parameter values are read correctly and that initializationPriorType is properly implemented. For the test cases, either the nominal values are directly verified, or, when testing start guesses that are randomly sampled, the mean and variance of multiple samples are evaluated.","category":"page"},{"location":"test_info.html#Hybrid-Models-Test","page":"Test Suite","title":"Hybrid Models Test","text":"","category":"section"},{"location":"test_info.html","page":"Test Suite","title":"Test Suite","text":"For each case, the following aspects are tested:","category":"page"},{"location":"test_info.html","page":"Test Suite","title":"Test Suite","text":"Correct evaluation of the model likelihood.\nAccuracy of simulated values when solving the model forward in time.\nGradient correctness. This is particularly important for SciML models, where computing gradients can be challenging.","category":"page"},{"location":"test_info.html","page":"Test Suite","title":"Test Suite","text":"If a tool supports providing the neural network model in a format other than the YAML format, we recommend modifying the tests in this folder to use another neural network format to verify the correctness of the implementation.","category":"page"},{"location":"format.html#Format-Specification","page":"Format","title":"Format Specification","text":"","category":"section"},{"location":"format.html","page":"Format","title":"Format","text":"A PEtab SciML problem extends the PEtab standard version 2 to accommodate hybrid models (SciML problems) that combine neural network and mechanistic components. Two new file types are introduced by the extension:","category":"page"},{"location":"format.html","page":"Format","title":"Format","text":"Neural Net File(s): Optional YAML file(s) describing neural net model(s).\nHybridization table: Table for assigning neural network outputs and inputs.","category":"page"},{"location":"format.html","page":"Format","title":"Format","text":"The extension further extends the following standard PEtab files:","category":"page"},{"location":"format.html","page":"Format","title":"Format","text":"Mapping Table: Extended to describe how neural network inputs and outputs map to PEtab variables.\nParameters Table: Extended to describe nominal values for network parameters.\nProblem YAML File: Extended to include a new SciML field for neural network models and (optionally) array or tensor formatted data.","category":"page"},{"location":"format.html","page":"Format","title":"Format","text":"All other PEtab files remain unchanged. This specification explains the format for each file that is added or modified by the PEtab SciML extension.","category":"page"},{"location":"format.html#hybrid_types","page":"Format","title":"High Level Overview","text":"","category":"section"},{"location":"format.html","page":"Format","title":"Format","text":"The PEtab SciML specification is designed to keep the dynamic model, neural network model, and PEtab problem as independent as possible while linking them through the hybridization and/or condition tables. In this context, mechanistic models are typically defined using community standards like SBML and are commonly simulated as systems of ordinary differential equations (ODEs), and here the terms mechanistic model and ODE are used interchangeably. Essentially, the PEtab SciML approach takes a PEtab problem involving a mechanistic ODE model and supports the integration of neural network inputs and outputs.","category":"page"},{"location":"format.html","page":"Format","title":"Format","text":"PEtab SciML supports two classes of hybrid models:","category":"page"},{"location":"format.html","page":"Format","title":"Format","text":"Pre simulation hybridization: The neural network model sets constant parameters and/or initial values in the ODE model prior to model simulation. Inputs are constant per simulation condition.\nIntra simulation hybridization: The neural network model appears in the ODE RHS and/or observable formula. Inputs are per time-point computed from simulated quantities.","category":"page"},{"location":"format.html","page":"Format","title":"Format","text":"A PEtab SciML problem can also include multiple neural networks. Aside from ensuring that neural networks do not conflict (e.g., by sharing the same output), no special considerations are required. Each additional network is included just as it would be in the single-network case.","category":"page"},{"location":"format.html#net_format","page":"Format","title":"Neural Network Model Format","text":"","category":"section"},{"location":"format.html","page":"Format","title":"Format","text":"The neural network model format is flexible, meaning models can be provided in any format compatible with the PEtab SciML format (for example, Lux.jl in PEtab.jl). Additionally, the petab_sciml library provides a neural network YAML file format that can be imported by tools across various programming languages. This format flexibility exists because, although the YAML format can accommodate many architectures, some may still be difficult to represent. However, the YAML format is recommended whenever possible to facilitate model exchange across different software.","category":"page"},{"location":"format.html","page":"Format","title":"Format","text":"A neural network model must consist of two parts to be compatible with the PEtab SciML standard:","category":"page"},{"location":"format.html","page":"Format","title":"Format","text":"layers: A constructor that defines the network layers, each with a unique identifier.\nforward: A forward pass function that, given input arguments, specifies the order in which layers are called, applies any activation functions, and returns one or several arrays. The forward function can accept more than one input argument (n > 1), and in the mapping table, the forward function's nth input argument (ignoring any potential class arguments such as self) is referred to as inputArgumentIndex{n}. Similar holds for the output. Aside from the neural network output values, every component that should be visible to other parts of the PEtab SciML problem must be defined elsewhere (e.g., in layers).","category":"page"},{"location":"format.html#hdf5_ps_structure","page":"Format","title":"Neural Network Parameters","text":"","category":"section"},{"location":"format.html","page":"Format","title":"Format","text":"All parameters for a neural network model are stored in an HDF5 file, with the file path specified in the problem YAML file. The HDF5 parameter file is expected to have the following structure for an arbitrary number of layers:","category":"page"},{"location":"format.html","page":"Format","title":"Format","text":"parameters.hdf5\n└───layerId{1} (group)\n│   ├── arrayId{1}\n│   └── arrayId{2}\n└───layerId{2} (group)\n    ├── arrayId{1}\n    └── arrayId{2}","category":"page"},{"location":"format.html","page":"Format","title":"Format","text":"The indexing convention and naming for arrayId depend on the neural network model library:","category":"page"},{"location":"format.html","page":"Format","title":"Format","text":"Neural network models in the PEtab SciML YAML format follow PyTorch indexing and naming conventions. For example, in a PyTorch linear layer, the arrays ids are weight and (optionally) bias\nNeural network models in other formats follow the indexing and naming conventions of the respective package and programming language.","category":"page"},{"location":"format.html#hdf5_input_structure","page":"Format","title":"Neural Network Input","text":"","category":"section"},{"location":"format.html","page":"Format","title":"Format","text":"Potential neural network input files are expected to have the following format:","category":"page"},{"location":"format.html","page":"Format","title":"Format","text":"input.hdf5\n└───input (group)\n    └─── input_array","category":"page"},{"location":"format.html","page":"Format","title":"Format","text":"As with parameters, the indexing depends on the neural network library:","category":"page"},{"location":"format.html","page":"Format","title":"Format","text":"Neural network models in the PEtab SciML YAML format follow PyTorch indexing. For example, if the first layer is Conv2d, the input should be in (C, W, H) format.\nNeural network models in other formats follow the indexing and naming conventions of the respective package and programming language.","category":"page"},{"location":"format.html","page":"Format","title":"Format","text":"tip: For developers: Respect memory order\nTools supporting the SciML extension should, for computational efficiency, reorder input data and potential layer parameter arrays to match the memory ordering of the target language. For example, PEtab.jl converts input data to follow Julia based indexing.","category":"page"},{"location":"format.html","page":"Format","title":"Format","text":"TODO: We will fix condition specific input in the YAML file later.","category":"page"},{"location":"format.html#YAML_net_format","page":"Format","title":"YAML Network file format","text":"","category":"section"},{"location":"format.html","page":"Format","title":"Format","text":"The petab_sciml library provides a YAML neural network file format for model exchange. The YAML format follows PyTorch conventions for layer names and arguments. While YAML files can be written manually, it is recommended approach to define a PyTorch nn.Module and use the petab_sciml library to automatically generate the YAML representation (see tutorials).","category":"page"},{"location":"format.html","page":"Format","title":"Format","text":"TODO: Should we have some description of the YAML format here? (can and should probably be added later)","category":"page"},{"location":"format.html#mapping_table","page":"Format","title":"Mapping Table","text":"","category":"section"},{"location":"format.html","page":"Format","title":"Format","text":"All neural networks are assigned an Id in the PEtab problem YAML file. A neural network Id is not considered a valid PEtab identifier, to prevent confusion regarding what its refers to (e.g., parameters, inputs, outputs). Consequently, every neural network input, parameter, and output referenced in the PEtab problem must be defined in the mapping table under modelEntityId and mapped to a PEtab identifier. Note, in this context array file Ids defined in the YAML file are considered valid entities in the PEtabEntityId column.","category":"page"},{"location":"format.html#modelEntityId-[STRING,-REQUIRED]","page":"Format","title":"modelEntityId [STRING, REQUIRED]","text":"","category":"section"},{"location":"format.html","page":"Format","title":"Format","text":"A modeling-language-independent syntax which refers to inputs, outputs, and parameters of neural networks.","category":"page"},{"location":"format.html#Parameters","page":"Format","title":"Parameters","text":"","category":"section"},{"location":"format.html","page":"Format","title":"Format","text":"The model Id $netId.parameters[$layerId].{[$arrayId]{[$parameterIndex]}} refers to the parameters of a neural network identified by $netId.","category":"page"},{"location":"format.html","page":"Format","title":"Format","text":"$layerId: The unique identifier of the layer (e.g., conv1).\n$arrayId: The parameter array name specific to that layer (e.g., weight).\n$parameterIndex: The indexing into the parameter array (syntax).","category":"page"},{"location":"format.html#Inputs","page":"Format","title":"Inputs","text":"","category":"section"},{"location":"format.html","page":"Format","title":"Format","text":"The model Id $netId.inputs{[$inputArgumentIndex]{[$inputIndex]}} refers to specific inputs of the network identified by $netId.","category":"page"},{"location":"format.html","page":"Format","title":"Format","text":"$inputArgumentIndex: The input argument number in the neural network forward function. Starts from 1.\n$inputIndex Indexing into the input argument (syntax). Should not be specified if the input is a file.","category":"page"},{"location":"format.html#Outputs","page":"Format","title":"Outputs","text":"","category":"section"},{"location":"format.html","page":"Format","title":"Format","text":"The model Id $netId.outputs{[outputArgumentIndex]{[$outputIndex]}} refers to specific outputs of a neural network identified by $netId.","category":"page"},{"location":"format.html","page":"Format","title":"Format","text":"$outputId: The output argument number in the neural network forward function. Starts from 1.\n$outputIndex: Indexing into the output argument (syntax)","category":"page"},{"location":"format.html#Nested-Identifiers","page":"Format","title":"Nested Identifiers","text":"","category":"section"},{"location":"format.html","page":"Format","title":"Format","text":"The PEtab SciML extension supports nested identifiers for mapping structured or hierarchical elements. Identifiers are expressed in a hierarchical which is indicate above using nested curly brackets. Valid examples are:","category":"page"},{"location":"format.html","page":"Format","title":"Format","text":"net1.parameters\nnet1.parameters[conv1]\nnet1.parameters[conv1].weight","category":"page"},{"location":"format.html","page":"Format","title":"Format","text":"warn: Do not break the hierarchy\nIdentifiers that break the hierarchy (e.g., net1.parameters.weight) are not valid.","category":"page"},{"location":"format.html#mapping_table_indexing","page":"Format","title":"Indexing","text":"","category":"section"},{"location":"format.html","page":"Format","title":"Format","text":"Indexing into arrays follows the format [i1, i2, ...], and indexing notation depends on the neural network library:","category":"page"},{"location":"format.html","page":"Format","title":"Format","text":"Neural network models in the PEtab SciML YAML format follow PyTorch indexing. Consequently, indexing is 0-based.\nNeural network models in other formats follow the indexing and naming conventions of the respective package and programming language.","category":"page"},{"location":"format.html#Assigning-Values","page":"Format","title":"Assigning Values","text":"","category":"section"},{"location":"format.html","page":"Format","title":"Format","text":"For assignments to nested PEtab identifiers (in the parameters, hybridisations, or conditions tables), assigned values must either:","category":"page"},{"location":"format.html","page":"Format","title":"Format","text":"Refer to another PEtab identifier with the same nested structure, or\nFollow the corresponding hierarchical HDF5 input or parameter structure.","category":"page"},{"location":"format.html#hybrid_table","page":"Format","title":"Hybridization Table","text":"","category":"section"},{"location":"format.html","page":"Format","title":"Format","text":"A tab-separated values file for assigning neural network inputs and outputs. Assignments in the table the table apply to all simulation conditions. Expected to have, in any order, the following two columns:","category":"page"},{"location":"format.html","page":"Format","title":"Format","text":"targetId targetValue\nNON_ESTIMATED_ENTITY_ID MATH_EXPRESSION\nnet1_input1 p1\nnet1_input2 p1\n... ...","category":"page"},{"location":"format.html#Detailed-Field-Description","page":"Format","title":"Detailed Field Description","text":"","category":"section"},{"location":"format.html","page":"Format","title":"Format","text":"targetId [NON_ESTIMATED_ENTITY_ID, REQUIRED]: The identifier of the non-estimated entity that will be modified. Restrictions depend on hybridization type (pre- or intra-simulation hybridization). See below.\ntargetValue [STRING, REQUIRED]: The value or expression that will be used to change the target.","category":"page"},{"location":"format.html#Pre-simulation-hybridization","page":"Format","title":"Pre-simulation hybridization","text":"","category":"section"},{"location":"format.html","page":"Format","title":"Format","text":"Pre-simulation neural network model inputs and outputs are constant targets (case 1 here).","category":"page"},{"location":"format.html#Inputs-2","page":"Format","title":"Inputs","text":"","category":"section"},{"location":"format.html","page":"Format","title":"Format","text":"Valid targetValue's for a neural network input are:","category":"page"},{"location":"format.html","page":"Format","title":"Format","text":"A parameter in the parameter table\nAn array input file (assigned an Id in the YAML problem file).","category":"page"},{"location":"format.html#Outputs-2","page":"Format","title":"Outputs","text":"","category":"section"},{"location":"format.html","page":"Format","title":"Format","text":"Valid targetId's for a neural network output are:","category":"page"},{"location":"format.html","page":"Format","title":"Format","text":"A non-estimated model parameter\nA specie's initial value (referenced by the specie's Id). In this case, any other specie initialization is overridden.","category":"page"},{"location":"format.html#Condition-and-Hybridization-Tables","page":"Format","title":"Condition and Hybridization Tables","text":"","category":"section"},{"location":"format.html","page":"Format","title":"Format","text":"Pre-simulation assignments can alternatively be made in the conditions table. Combinations are also permitted, for example, all inputs can be assigned in the condition table while all outputs are assigned in the hybridization table. Importantly, however, since the hybridization table defines assignments for all simulation conditions, any targetId assigned in the condition table cannot be assigned in the hybridization table, and vice versa.","category":"page"},{"location":"format.html#Intra-simulation-hybridization","page":"Format","title":"Intra-simulation hybridization","text":"","category":"section"},{"location":"format.html","page":"Format","title":"Format","text":"Intra-simulation neural network models depend on model simulated model quantities (case 2 here).","category":"page"},{"location":"format.html#Inputs-3","page":"Format","title":"Inputs","text":"","category":"section"},{"location":"format.html","page":"Format","title":"Format","text":"Valid targetValue for a neural network input is an expression that depend on model species, time, and/or parameters. Any species and/or parameters in the expression are expected to be evaluated at the given time-value.","category":"page"},{"location":"format.html#Outputs-3","page":"Format","title":"Outputs","text":"","category":"section"},{"location":"format.html","page":"Format","title":"Format","text":"Valid targetId for a neural network output is a constant model parameter. During PEtab problem import, assigned parameters are replaced by the neural network output in the ODE RHS.","category":"page"},{"location":"format.html","page":"Format","title":"Format","text":"Importantly, it is invalid for a model parameter assigned by a neural network output to appear in both the ODE RHS and the observable formula. In the observable formula, neural network output PEtab identifiers are expected to be directly encoded, making a neural network–assigned parameter in the formula ambiguous. Conversely, it is considered valid if a neural network output is used to assign an ODE model parameter and is also directly encoded in the observable formula.","category":"page"},{"location":"format.html#parameter_table","page":"Format","title":"Parameter Table","text":"","category":"section"},{"location":"format.html","page":"Format","title":"Format","text":"The parameter table follows the same format as in PEtab version 2, with a subset of fields extended to accommodate neural network parameters. This section focuses on columns extended by the SciML extension.","category":"page"},{"location":"format.html","page":"Format","title":"Format","text":"note: Specific Assignments Have Precedence\nMore specific assignments (e.g., netId.parameters[layerId] instead of netId.parameters) have precedence for nominal values, priors, and other setting. For example, if a nominal values is assigned to netId and a different nominal value is assigned to netId.parameters[layerId], the latter is used in place of the former.","category":"page"},{"location":"format.html#Detailed-Field-Description-2","page":"Format","title":"Detailed Field Description","text":"","category":"section"},{"location":"format.html","page":"Format","title":"Format","text":"parameterId [String, REQUIRED]: The neural network or a specific layer/parameter array id. The target of the parameterId must be assigned via the mapping table.\nnominalValue [String | NUMERIC, REQUIRED]: Neural network nominal values. This can be:\nA PEtab variable that via the problem YAML file corresponds to an HDF5 file with the required structure. If no file exists at the given path when the problem is imported and the parameters are set to be estimated, a file is created with randomly sampled values. Unless a numeric value is provided, referring to the same file is required for all assignments for a neural network, since all neural network parameters should be collected in a single HDF5 file following the structure described here.\nA numeric value applied to all parameters under parameterId.\nestimate [0 | 1, REQUIRED]: Indicates whether the parameters are estimated (1) or fixed (0). This must be consistent across layers. For example, if netId.parameters has estimate = 0, then potential layer rows must also be 0. In other words, freezing individual network parameters is not allowed.","category":"page"},{"location":"format.html#Bounds-for-neural-net-parameters","page":"Format","title":"Bounds for neural net parameters","text":"","category":"section"},{"location":"format.html","page":"Format","title":"Format","text":"Bounds can be specified for an entire network or its nested identifiers. However, most optimization algorithms used for neural networks, such as ADAM, do not support parameter bounds in their standard implementations. Therefore, neural net bounds are optional and default to -inf for the lower bound and inf for the upper bound.","category":"page"},{"location":"format.html#YAML_file","page":"Format","title":"Problem YAML File","text":"","category":"section"},{"location":"format.html","page":"Format","title":"Format","text":"An extension section is included in the PEtab SciML YAML file for specifying neural network YAML files, as well as array parameter, input, and output files. These elements are defined using the following key-value mappings:","category":"page"},{"location":"format.html","page":"Format","title":"Format","text":"file[extensions][petab_sciml][neural_nets]: Neural network models. Here, each network is defined as a key-value mapping, where the key is the unique neural network Id (netId), and the corresponding value is another key-value mapping:\n[netId][location]: The file path where the neural network model is stored.\n[netId][format]: The neural network format. Expected to be YAML if the network is provided in the PEtab SciML library YAML format. Otherwise, the neural network library should be provided (e.g Lux.jl or equinox.py).\n[netId][hybridization]: The neural network hybridization type. Expected to be either pre_simulation or intra_simulation (for type information see here).\nyaml_file[extensions][petab_sciml][array_files]  Potential array files. Parameter files are expected to follow the structure described here, and input files should follow the structure described here. Each entry is defined with a key-value mapping where the key is the array Id (arrayId), and the corresponding value is another key-value mapping:\n[arrayId][location]: The file path.\n[arrayId][format]: The file format (e.g., HDF5).","category":"page"},{"location":"format.html","page":"Format","title":"Format","text":"If a neural network is provided in another format than the YAML format, respective tool must provide the network during problem import. Note that regardless of neural-network format, for exchange purposes the neural network model must be available in a file (not in the main script).","category":"page"},{"location":"index.html#PEtab-SciML-Extension","page":"Home","title":"PEtab SciML Extension","text":"","category":"section"},{"location":"index.html","page":"Home","title":"Home","text":"The PEtab SciML extension expands the PEtab standard for parameter estimation problems to support hybrid models that combine data-driven neural network models with a mechanistic Ordinary Differential Equation (ODE) model. This enables a reproducible format for specifying and ultimately fitting hybrid models to time-series or dose-response data. This repository contains both the format specification and a Python library for exporting neural network models to a standard YAML format, which can be imported across multiple programming languages.","category":"page"},{"location":"index.html#Highlights","page":"Home","title":"Highlights","text":"","category":"section"},{"location":"index.html","page":"Home","title":"Home","text":"A file format that supports three approaches for combining mechanistic and neural network models:\nIncorporating neural network model(s) data-driven model in the ODE model right-hand side and/or observable formula which describes the mapping between simulation output and measurement data.\nIncorporating neural network model(s) to set constant model parameter values prior to simulation, allowing for example, available metadata to be used to set parameter values.\nSupport for many neural network architectures, including most standard layers and activation functions available in packages such as PyTorch.\nImplementations in tools across two programming languages. In particular, both PEtab.jl in Julia and AMICI in Python (Jax) can import problems in the PEtab SciML format.\nAn extensive test suite that ensures the correctness of tools supporting the format.","category":"page"},{"location":"index.html#Installation","page":"Home","title":"Installation","text":"","category":"section"},{"location":"index.html","page":"Home","title":"Home","text":"Currently, an installation of git and a recent version of Python 3 is required. As usual, a Python virtual environment is encouraged.","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"pip install git+https://github.com/sebapersson/petab_sciml#subdirectory=src/python&egg=petab_sciml","category":"page"},{"location":"index.html#Getting-help","page":"Home","title":"Getting help","text":"","category":"section"},{"location":"index.html","page":"Home","title":"Home","text":"If you have any problems with either using this package, or with creating a PEtab SciML problem, here are some helpful tips:","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"Please open an issue on GitHub.\nPost your questions in the #sciml-sysbio channel on the Julia Slack. While this is not a Julia package, the developers are active on that forum.","category":"page"},{"location":"tutorial.html#Tutorial","page":"Tutorial","title":"Tutorial","text":"","category":"section"},{"location":"tutorial.html","page":"Tutorial","title":"Tutorial","text":"The tutorials will consist of:","category":"page"},{"location":"tutorial.html","page":"Tutorial","title":"Tutorial","text":"Overarching tutorial where we show how to implement the Lotka-Volterra problem from the UDE paper.\nExtended tutorial 1 where we have a neural-network setting parameters, here it is also worthwhile to consider simulation conditions.\nExtended tutorial 2 where we have a neural network in the observable function.","category":"page"},{"location":"tutorial.html","page":"Tutorial","title":"Tutorial","text":"If time, add tutorial for having two neural networks.","category":"page"}]
}
