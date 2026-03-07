"""
Baseline NKI softmax kernel for AWS Trainium/Trainium2.

Computes numerically-stable softmax along the free dimension (axis=1)
using NKI ISA-level APIs with partition-dimension tiling.

This standalone kernel serves as the non-fused baseline to compare
against the fused GEMM + Softmax kernel.
"""

import numpy as np
from neuronxcc import nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl


def numpy_softmax(x):
    """NumPy reference: softmax along last axis."""
    m = np.max(x, axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=-1, keepdims=True)


@nki.jit
def softmax_kernel(x_ref):
    """
    Numerically-stable softmax with tiling along partition dimension.

    Input:  (M, N) in HBM
      - M (partition dim) is tiled in chunks of PMAX (128)
      - N (free dim) is the softmax reduction axis, chunked by FMAX (512)

    Output: (M, N) in HBM

    Constraints:
      - M % 128 == 0
      - N % 512 == 0
    """
    M, N = x_ref.shape
    PMAX = nl.tile_size.pmax
    FMAX = nl.tile_size.gemm_moving_fmax

    assert M % PMAX == 0
    assert N % FMAX == 0

    out_ref = nl.ndarray((M, N), dtype=x_ref.dtype, buffer=nl.shared_hbm)
    n_tiles = M // PMAX
    n_chunks = N // FMAX

    for i in nl.affine_range(n_tiles):
        # Load (PMAX, N) tile from HBM into SBUF
        x_tile = nl.ndarray(
            (nl.par_dim(PMAX), N), dtype=nl.float32, buffer=nl.sbuf)
        x_tile[...] = nl.load(x_ref[nl.ds(i * PMAX, PMAX), :])

        # ---- Row-wise max for numerical stability ----
        # Partial max per FMAX chunk, then global max (negated for activation bias)
        partial_max = nl.ndarray(
            (nl.par_dim(PMAX), n_chunks), dtype=nl.float32, buffer=nl.sbuf)
        for j in nl.affine_range(n_chunks):
            partial_max[:, j] = nisa.tensor_reduce(
                op=nl.max,
                data=x_tile[:, nl.ds(j * FMAX, FMAX)],
                axis=1)
        neg_max = nisa.tensor_reduce(
            op=nl.max, data=partial_max, axis=1, negate=True)

        # ---- exp(x - max) and partial row sums ----
        exp_tile = nl.ndarray(
            (nl.par_dim(PMAX), N), dtype=nl.float32, buffer=nl.sbuf)
        partial_sum = nl.ndarray(
            (nl.par_dim(PMAX), n_chunks), dtype=nl.float32, buffer=nl.sbuf)
        for j in nl.affine_range(n_chunks):
            # activation computes exp(scale * data + bias) = exp(x + (-max))
            exp_tile[:, nl.ds(j * FMAX, FMAX)] = nisa.activation(
                op=nl.exp,
                data=x_tile[:, nl.ds(j * FMAX, FMAX)],
                bias=neg_max, scale=1.0)
            partial_sum[:, j] = nisa.tensor_reduce(
                op=nl.add,
                data=exp_tile[:, nl.ds(j * FMAX, FMAX)],
                axis=1)

        # ---- Normalize: exp(x - max) / row_sum ----
        row_sum = nisa.tensor_reduce(op=nl.add, data=partial_sum, axis=1)
        inv_sum = nisa.reciprocal(data=row_sum)

        result = nl.ndarray(
            (nl.par_dim(PMAX), N), dtype=x_ref.dtype, buffer=nl.sbuf)
        for j in nl.affine_range(n_chunks):
            result[:, nl.ds(j * FMAX, FMAX)] = nisa.tensor_scalar(
                data=exp_tile[:, nl.ds(j * FMAX, FMAX)],
                op0=nl.multiply, operand0=inv_sum,
                engine=nisa.vector_engine,
                dtype=x_ref.dtype)

        nl.store(dst=out_ref[nl.ds(i * PMAX, PMAX), :], value=result)

    return out_ref


# ---------------------------------------------------------------------------
# Testing (requires Trainium instance)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    M, N = 1024, 4096
    x = np.random.randn(M, N).astype(np.float32)

    expected = numpy_softmax(x)
    actual = softmax_kernel(x)

    if np.allclose(actual, expected, atol=1e-5, rtol=1e-3):
        print("PASS: softmax_kernel matches NumPy reference")
    else:
        max_err = np.max(np.abs(actual - expected))
        print(f"FAIL: max abs error = {max_err}")
