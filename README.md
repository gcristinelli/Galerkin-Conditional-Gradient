# Galerkin-Conditional-Gradient
[![DOI](https://zenodo.org/badge/710899648.svg)](https://zenodo.org/doi/10.5281/zenodo.10048384)

This module solves the control problem with Total Variation regularization

min_{u\in BV(\Omega)} 1/2 |Ku-y0|^2 + alpha TV(u,\Omega)

where K is the control-to-state operator associated with a Poisson-type PDE.

It uses the Dinkelbach-FCGCG approach described in the paper "Conditional gradient by graph cuts for total variation regularization with PDE constraints".

Important libraries:

FEniCS (Dolfin) --version 2019.1.0 (https://fenicsproject.org/) 
Maxflow (http://pmneila.github.io/PyMaxflow/maxflow.html)
