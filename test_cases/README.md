# PEtab SciML Extension Test Cases

This directory contains the test cases for the PEtab SciML extension. For each case, the following things are tested:

* That the data-driven (typically neural network) model evaluates correctly. This means that any importer should support the property to separately compute the data-driven model for a given input.
* That the model likelihood evaluates correctly.
* That the simulated values are correct from solving the model forward in time.
* Gradient correctness. Especially with SciML models, computing the gradient can be challenging as deriving the Jacobian of the ODE model can be tricky. For example, without an automatic differentiation framework it can be hard to differentiate data-driven models that are not a part of the ODE model.

Below is a very brief summary of each case; more details can be found in the corresponding directory.

## Case 001

A feed-forward neural network is a part of the ODE model.

## Case 002

A feed-forward neural network sets one of the derivatives of the ODE model.

## Case 003

A feed-forward neural network sets the values for a subset of ODE model kinetic parameters. As the neural network is not part of the ODE right-hand side, it only needs to be evaluated once per likelihood computation.

## Case 004

A feed-forward neural network sets the values for a subset of ODE model kinetic parameters, where the input to the neural network depends on the simulation condition.

## Case 005

A feed-forward neural network appears only in one of the observable formulas.
