# MPO Small Systems

This folder contains a notebook focused on **Matrix Product Operator (MPO)** constructions and validation on **small systems**, where the tensor-based representations can be checked more explicitly against the original operators.

## Contents

- `Proyecto_MPO_Small_Systems.ipynb`

## Overview

The main goal of this notebook is to study operator representations in settings where the full matrix can still be inspected directly. Working with smaller systems makes it possible to:

- compare exact and tensor-based representations more rigorously,
- validate MPO constructions explicitly,
- inspect sparsity patterns and operator structure,
- and better understand how bond dimensions relate to the underlying matrix organization.

This notebook plays an important role in the repository because it complements the larger-scale and more exploratory notebooks with a more controlled validation setting.

## Main topics

The notebook includes aspects such as:

- construction of small Hamiltonians and related operators,
- conversion of structured matrices into tensor-network-friendly forms,
- MPO representations,
- explicit reconstruction and comparison tests,
- numerical checks of errors and consistency,
- and interpretation of bond dimensions in simple cases.

## Why this notebook matters

Small systems are especially useful for validation.  
They provide a setting where tensor constructions are not only implemented, but also **checked carefully** against known results.

For that reason, this notebook is one of the most important pieces in the repository from the point of view of **numerical reliability** and **methodological clarity**.

## Role within the repository

If `ProjectQTCI.ipynb` is a good first notebook to understand the overall direction of the project, this notebook is a strong companion piece for understanding how the operator constructions are validated in practice.
