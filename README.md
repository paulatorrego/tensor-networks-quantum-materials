# Tensor Methods for Quantum Materials

This repository gathers the main computational work I developed in Julia on **tensor methods for quantum materials**, with a particular focus on **quantics representations**, **tensor trains**, **matrix product operators (MPOs)**, and **tensor cross interpolation (TCI)**.

The notebooks collected here explore structured representations of Hamiltonians and related operators, combining ideas from computational physics, tensor networks, and scientific computing. The overall goal is to study how large, structured operators arising in quantum-materials models can be represented, compressed, analyzed, and validated using tensor-based approaches.

## Project context

This repository was developed in the context of my **JAE Intro research fellowship**, where I worked on computational tools and tensor-based methods for quantum materials. The project combines:

- physical modelling of structured quantum systems,
- numerical experimentation in Julia,
- tensor decompositions and quantics encodings,
- and validation of operator representations on both small and larger systems.

Rather than being a collection of unrelated notebooks, this repository is intended as a coherent record of that research workflow.

## Main topics

The main themes explored in this repository are:

- **Quantics representations** of vectors and operators
- **Tensor-train (TT / QTT)** descriptions of structured objects
- **Matrix Product Operator (MPO)** constructions
- **Tensor Cross Interpolation (TCI)** as a data-efficient tensor approximation strategy
- Study of **bond dimensions**, sparsity structure, and numerical validation
- Applications to **Hamiltonians and trial operators** motivated by quantum-materials models

## Repository structure

```text
.
├── README.md
├── Project.toml
├── Manifest.toml
├── notebooks/
│   ├── ProjectQTCI.ipynb
│   ├── Proyecto_MPO_Small_Systems.ipynb
│   ├── QTCI_completo.ipynb
│   └── TareaLimpio.ipynb
└── references/
