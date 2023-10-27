# Galerkin-Conditional-Gradient

This module solves the control problem with Total Variation regularization

min_{u\in BV(\Omega)} 1/2 |Ku-y0|^2 + alpha TV(u,\Omega)

where K is the control-to-state operator associated with a Poisson-type PDE.

It uses the Dinkelbach-FCGCG approach described in the paper "Sparse optimization for discretized PDE-constrained problems with total variation regularization".

Important libraries:

FEniCS (Dolfin) --version 2019.1.0 (https://fenicsproject.org/) 
Maxflow (http://pmneila.github.io/PyMaxflow/maxflow.html)
