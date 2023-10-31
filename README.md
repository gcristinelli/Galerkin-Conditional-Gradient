# Galerkin-Conditional-Gradient
[![DOI](https://zenodo.org/badge/710899648.svg)](https://zenodo.org/doi/10.5281/zenodo.10048384)

Authors: Giacomo Cristinelli, José A. Iglesias, Daniel Walter

This module solves the control problem with Total Variation regularization

$$min_{u\in BV(\Omega)} 1/2 |Ku-y_o|^2 + \alpha TV(u,\Omega)$$

where K is the control-to-state operator associated with a Poisson-type PDE.

It employs the method described in the paper "Conditional gradients for total variation regularization with PDE constraints: a graph cuts approach". 
Preprint available at: https://arxiv.org/abs/2310.19777

Important libraries:

FEniCS (Dolfin) --version 2019.1.0 (https://fenicsproject.org/) 

Maxflow (http://pmneila.github.io/PyMaxflow/maxflow.html)
