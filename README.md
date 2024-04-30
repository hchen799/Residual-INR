# Residual-INR: Communication Efficient Fog Online Learning Using Implicit Neural Representation

## Introduction

This repository includes all of the code implementation for the Residual-INR submitted to ICCAD'24.

## Environment Setup

- **Software**: All of our experiments are finihsed with Python. You can set up the Python environment by running the following command:

```bash
pip install -r requirements.txt
```

- **Hardware**: INR decoding on GPU is finished on NVIDIA RTX A6000 (48GB). You can run our code on any GPU platforms that can support PyTorch.

## Residual-INR Encoding

#### Background INR Encoding

Background INR encoding is based on Rapid-INR using MLPs. Run the following commands to encode a dataset:

```bash
cd B_INR_encode
python background_INR_encoding.py --dataset_dir ../data/OTB/
```
The dataset path is specified as the --dataset_dir flag. 

