# PEtab SciML Extension

The PEtab SciML extension expands the [PEtab](https://petab.readthedocs.io/en/latest/) standard for parameter estimation problems to support hybrid models that combine data-driven neural network models with a mechanistic Ordinary Differential Equation (ODE) model. This enables a reproducible format for specifying and ultimately fitting hybrid models to time-series data. This repository contains both the format specification and a Python library for exporting neural network models to a standard YAML format, which can be imported across multiple programming languages.

## Major Highlights

* A file format that supports three approaches for combining mechanistic and neural network models:
  * Incorporating neural network model(s) data-driven model in the ODE model right-hand side.
  * Incorporating neural network model(s) in the observable formula which describes the mapping between simulation output and measurement data.
  * Incorporating neural network model(s) to set constant model parameter values prior to simulation, allowing for example, available metadata to be used to set parameter values.
* support for many neural network architectures, including most standard layers and activation functions available in packages such as PyTorch.
* implementations in tools across two programming languages. In particular, both [PEtab.jl](https://github.com/sebapersson/PEtab.jl) in Julia and AMICI in Python (Jax) can import problems in the PEtab SciML format.
* an extensive test suite that ensures the correctness of tools supporting the format.

## Installation

Currently, an installation of git and a recent version of Python 3 is required. As usual, a Python virtual environment is encouraged.

```bash
pip install git+https://github.com/sebapersson/petab_sciml#subdirectory=src/python&egg=petab_sciml
```

## Getting help

If you have any problems with either using this package, or with creating a PEtab SciML problem, here are some helpful tips:

* Please open an issue on [GitHub](https://github.com/sebapersson/petab_sciml/issues).
* Post your questions in the `#sciml-sysbio` channel on the [Julia Slack](https://julialang.org/slack/). While this is not a Julia package, the developers are active on that forum.
