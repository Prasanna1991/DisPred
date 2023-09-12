# DisPred
Code for running DisPred algorithm
Please cite it as : [![DOI](https://zenodo.org/badge/517768793.svg)](https://zenodo.org/badge/latestdoi/517768793)

## Repo Contents

run.py ==> to run DisPred algorithm
data.py, models.py, utils.py ==> supporting code for data, model and utility functions. 

## System Requirements

Hardware requirements

This package requires only a standard computer with about 4 GB of RAM. For optimal performance, we recommend a computer with the following hardware properties:

RAM: 4+ GB
CPU: 4+ cores, 3.3+ GHz
GPU: 16 GB

We examined the codes on both local computer and remote and with and without GPUs.

Software requirements:
PyTorch (1.7.0)
Numpy (1.19.5)
Pandas (1.1.5)
Scikit-learn (0.24.2)

pip3 install torch==1.7.0

## Example:
Please keep the dataset in the associated folder and then, 

python3 run.py

## Citations

The paper titled by "Improving genetic risk prediction across diverse population by disentangling ancestry representations" is uploaded to Arxiv:

https://arxiv.org/pdf/2205.04673

Please feel free to write me your questions at: pgyawali@stanford.edu




