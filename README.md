# Trainium Kernel Fusion with NKI

## Overview
This project explores kernel fusion for Transformer inference on AWS Trainium
using the Neuron Kernel Interface (NKI). We implement fused GEMM + activation
kernels to reduce HBM traffic by keeping intermediate results on-chip in SBUF.

The focus is on two performance-critical Transformer components:
- MLP block: fused GEMM + GELU
- Attention block: fused QKᵀ GEMM + Softmax

## Key Techniques
- GEMM epilogue fusion using NKI
- Explicit SBUF tiling and DMA orchestration
- Piecewise-linear (PWL) and Taylor-series activation approximations
- Precision and quantization analysis (FP32, BF16)

## Hardware Target
- AWS Trainium / Trainium2 (`trn1`, `trn2`)
- NKI kernels in this repo are written with `nl.tile_size` values and run on
  NeuronCore-v2/v3, but for `trn2` you should use a recent Neuron SDK release.

## TRN2 Notes
- NKI supports `trn2`; update Neuron SDK to a version that includes Trainium2
  NKI support.
- If you want to force compile target, set:
  `NEURON_CC_FLAGS="--target=trn2"`
- You can also run:
  `python benchmarks/attention_benchmark.py --mode benchmark --target trn2`
  or:
  `bash scripts/compile_kernels.sh --target trn2`
- Vector Engine + Scratchpad Buffer (SBUF)

## Evaluation
Fused kernels are compared against non-fused NKI baselines using:
- End-to-end latency
- Throughput
- HBM read/write traffic
- Numerical error from approximation and quantization

Profiling is performed with `neuron-profile` to validate reduced memory traffic
and improved pipeline overlap.

## Repository Structure
- `kernels/` – Baseline and fused NKI kernels
- `approximations/` – PWL LUTs and Taylor-series implementations
- `quantization/` – Precision and mixed-precision experiments
- `benchmarks/` – Microbenchmarks and Transformer block evaluation
- `profiling/` – Neuron profiling outputs and analysis
- `report/` – Final report and figures


Authors: Allen Chau, Julian Allchin
