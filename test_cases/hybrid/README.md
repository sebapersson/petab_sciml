# Hybrid tests

These tests check that hybrid models—combining mechanistic and data-driven components—are imported correctly. Note that data-driven models can be integrated in several ways: directly in the ODE right-hand side, used before the ODE (e.g., for parameter value settings), or included in the observable function. More information on each test can be found in its respective README.

For each case, the following things are tested:

* That the model likelihood evaluates correctly.
* That the simulated values are correct from solving the model forward in time.
* Gradient correctness. Especially with SciML models, computing the gradient can be challenging as computing the Jacobian of the ODE model can be tricky, or to get the gradient of a neural network that sets model parameters.

All reference solutions are computed using high-order (order 9) ODE-solvers with small (1e-12) tolerances.
