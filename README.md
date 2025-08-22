# Multiscale Modeling of Cell Fate Decisions: Integrating Gene Regulatory Networks with Population Dynamics via Physics-Informed Neural Networks

![Overview](Overview.png)


## Getting Started
OPCT takes the input of time-series scRNA-seq data and outputs:
- infered the cell velocity in continuous time;
- infered the cell trajectory in continuous time;
- infered the cell growth;
- derived the internal gene regulatory network of the cell;
- acquired the latent time of the cell;
- acquired the perturbation trajectory.


## Requirements and Setup

```
conda create -n OPCT python=3.8.10
conda activate OPCT
pip install -r requirements.txt
```
