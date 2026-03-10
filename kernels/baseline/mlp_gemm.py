"""
Baseline (non-fused) MLP kernels for AWS Trainium using NKI.

Implements the MLP forward pass as two separate kernel launches:
  1. gemm_kernel:  Y = X @ W + bias   (writes intermediate Y to HBM)
  2. gelu_kernel:  Z = GELU(Y)        (reads Y from HBM, writes Z to HBM)

The intermediate tensor Y takes a full HBM round-trip between the two
kernels. This is the memory-wall bottleneck that kernel fusion eliminates.
"""

import nki
import nki.language as nl
import numpy as np

# Tile sizes for Neuron Core v2
TILE_M = nl.tile_size.gemm_stationary_fmax  # 128 — LHS free dimension
TILE_K = nl.tile_size.pmax                  # 128 — contraction / partition dim
TILE_N = nl.tile_size.gemm_moving_fmax      # 512 — RHS free dimension


@nki.jit
def gemm_kernel(lhsT, rhs, bias):
    """Tiled GEMM with bias:  result = lhsT^T @ rhs + bias

    Parameters
    ----------
    lhsT  : tensor [K, M]   pre-transposed input activations
    rhs   : tensor [K, N]   weight matrix
    bias  : tensor [1, N]   bias vector (broadcast along M)

    Returns
    -------
    result : tensor [M, N]  written to HBM
    """
    K, M = lhsT.shape
    K_, N = rhs.shape
    assert K == K_
    assert M % TILE_M == 0 and N % TILE_N == 0 and K % TILE_K == 0

    result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

    for m in nl.affine_range(M // TILE_M):
        for n in nl.affine_range(N // TILE_N):
            # FP32 accumulator in PSUM for numerical stability
            acc = nl.zeros((TILE_M, TILE_N), nl.float32, buffer=nl.psum)

            for k in nl.affine_range(K // TILE_K):
                # HBM → SBUF
                lhsT_tile = nl.load(
                    lhsT[k * TILE_K:(k + 1) * TILE_K,
                         m * TILE_M:(m + 1) * TILE_M])
                rhs_tile = nl.load(
                    rhs[k * TILE_K:(k + 1) * TILE_K,
                        n * TILE_N:(n + 1) * TILE_N])

                # Partial matmul: (lhsT_tile)^T @ rhs_tile, accumulated in PSUM
                acc += nl.matmul(lhsT_tile, rhs_tile, transpose_x=True)

            # PSUM → SBUF with dtype cast
            acc_sb = nl.copy(acc, dtype=result.dtype)

            # Broadcast-add bias: [1, TILE_N] → [TILE_M, TILE_N]
            bias_tile = nl.load(bias[0:1, n * TILE_N:(n + 1) * TILE_N])
            acc_sb = acc_sb + bias_tile

            # Write intermediate result to HBM
            nl.store(
                result[m * TILE_M:(m + 1) * TILE_M,
                       n * TILE_N:(n + 1) * TILE_N],
                value=acc_sb)

    return result


@nki.jit
def gelu_kernel(y):
    """Element-wise GELU activation.

    Reads the intermediate GEMM output from HBM, applies exact GELU
    on the Vector Engine, and writes the result back to HBM.

    Parameters
    ----------
    y : tensor [M, N]  input (read from HBM)

    Returns
    -------
    z : tensor [M, N]  GELU(y), written to HBM
    """
    M, N = y.shape
    assert M % TILE_M == 0 and N % TILE_N == 0

    z = nl.ndarray((M, N), dtype=y.dtype, buffer=nl.shared_hbm)

    for m in nl.affine_range(M // TILE_M):
        for n in nl.affine_range(N // TILE_N):
            tile = nl.load(
                y[m * TILE_M:(m + 1) * TILE_M,
                  n * TILE_N:(n + 1) * TILE_N])
            activated = nl.gelu(tile)
            nl.store(
                z[m * TILE_M:(m + 1) * TILE_M,
                  n * TILE_N:(n + 1) * TILE_N],
                value=activated)

    return z


def mlp_baseline(X, W, bias):
    """Non-fused MLP forward pass.

    Executes GEMM and GELU as separate kernel launches. The intermediate
    tensor Y is materialized in HBM between the two kernels — this is
    the memory-wall overhead that the fused kernel will eliminate.

    Parameters
    ----------
    X    : ndarray [M, K]  input activations
    W    : ndarray [K, N]  weight matrix
    bias : ndarray [N]     bias vector

    Returns
    -------
    Z : ndarray [M, N]     GELU(X @ W + bias)
    """
    lhsT = X.T.copy()
    bias_2d = bias.reshape(1, -1)

    # Kernel launch 1: GEMM — writes Y to HBM
    Y = gemm_kernel(lhsT, W, bias_2d)

    # Kernel launch 2: GELU — reads Y from HBM, writes Z to HBM
    Z = gelu_kernel(Y)

    return Z


# ── NumPy reference for correctness verification ────────────────────
def mlp_reference(X, W, bias):
    """NumPy golden reference: GELU(X @ W + bias)"""
    from scipy.special import erf
    Y = X @ W + bias
    return 0.5 * Y * (1.0 + erf(Y / np.sqrt(2.0)))


if __name__ == "__main__":
    # Quick sanity check (dimensions must be multiples of tile sizes)
    M, K, N = 256, 512, 1024
    X = np.random.randn(M, K).astype(np.float32)
    W = np.random.randn(K, N).astype(np.float32)
    bias = np.random.randn(N).astype(np.float32)

    Z = mlp_baseline(X, W, bias)
    Z_ref = mlp_reference(X, W, bias)

    max_err = np.max(np.abs(Z - Z_ref))
    print(f"Shape: X={X.shape} W={W.shape} → Z={Z.shape}")
    print(f"Max absolute error vs NumPy reference: {max_err:.6e}")
