"""
Fused GEMM + GELU MLP kernel for AWS Trainium using NKI.

Computes: Z = GELU(X @ W + bias) in a single kernel launch.

Unlike the baseline which writes the intermediate Y = X @ W + bias to HBM
and then reads it back for GELU, this kernel applies GELU on-chip in SBUF
immediately after the GEMM accumulation — eliminating the HBM round-trip.

Two GELU approximation modes:
  - PWL:    Hardware piecewise-linear via nl.gelu (Vector Engine lookup table)
  - Taylor: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            using sigmoid identity: tanh(z) = 2*sigmoid(2z) - 1
"""

import numpy as np
from neuronxcc import nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa

# Tile sizes for Neuron Core v2 (must match baseline)
TILE_M = nl.tile_size.gemm_stationary_fmax  # 128 — LHS free dimension
TILE_K = nl.tile_size.pmax                  # 128 — contraction / partition dim
TILE_N = nl.tile_size.gemm_moving_fmax      # 512 — RHS free dimension


def _taylor_gelu(x):
    """Taylor-approximate GELU on an SBUF tile.

    GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

    Uses the identity tanh(z) = 2 * sigmoid(2z) - 1 to leverage the
    Vector Engine's hardware sigmoid unit.
    """
    SQRT_2_OVER_PI = 0.7978845608028654
    COEFF = 0.044715

    # x^3 = x * x * x
    x_sq = x * x
    x_cubed = x_sq * x

    # inner = sqrt(2/pi) * (x + 0.044715 * x^3)
    inner = nl.multiply(x + COEFF * x_cubed, SQRT_2_OVER_PI)

    # tanh(inner) = 2 * sigmoid(2 * inner) - 1
    sigmoid_2z = nisa.activation(op=nl.sigmoid, data=inner, scale=2.0)
    tanh_val = sigmoid_2z * 2.0 - 1.0

    # GELU = 0.5 * x * (1 + tanh_val)
    return 0.5 * x * (1.0 + tanh_val)


@nki.jit
def fused_mlp_pwl(lhsT, rhs, bias):
    """Fused GEMM + GELU (hardware PWL): Z = GELU(lhsT^T @ rhs + bias)

    GELU is applied on-chip via the Vector Engine's piecewise-linear unit.
    No intermediate tensor is written to HBM.

    Parameters
    ----------
    lhsT : tensor [K, M]  pre-transposed input activations
    rhs  : tensor [K, N]  weight matrix
    bias : tensor [1, N]  bias vector (broadcast along M)

    Returns
    -------
    result : tensor [M, N]
    """
    K, M = lhsT.shape
    K_, N = rhs.shape
    assert K == K_
    assert M % TILE_M == 0 and N % TILE_N == 0 and K % TILE_K == 0

    result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

    for m in nl.affine_range(M // TILE_M):
        for n in nl.affine_range(N // TILE_N):
            # FP32 accumulator in PSUM
            acc = nl.zeros((TILE_M, TILE_N), nl.float32, buffer=nl.psum)

            for k in nl.affine_range(K // TILE_K):
                lhsT_tile = nl.load(
                    lhsT[k * TILE_K:(k + 1) * TILE_K,
                         m * TILE_M:(m + 1) * TILE_M])
                rhs_tile = nl.load(
                    rhs[k * TILE_K:(k + 1) * TILE_K,
                        n * TILE_N:(n + 1) * TILE_N])
                acc += nl.matmul(lhsT_tile, rhs_tile, transpose_x=True)

            # PSUM -> SBUF
            acc_sb = nl.copy(acc, dtype=result.dtype)

            # Bias add (on-chip) — broadcast along partition dim
            bias_tile = nl.load(bias[0:1, n * TILE_N:(n + 1) * TILE_N])
            acc_sb = acc_sb + bias_tile.broadcast_to((TILE_M, TILE_N))

            # GELU via hardware PWL (on-chip, no HBM round-trip)
            activated = nl.gelu(acc_sb)

            # Single HBM write of final result
            nl.store(
                result[m * TILE_M:(m + 1) * TILE_M,
                       n * TILE_N:(n + 1) * TILE_N],
                value=activated)

    return result


@nki.jit
def fused_mlp_taylor(lhsT, rhs, bias):
    """Fused GEMM + GELU (Taylor approximation): Z = GELU(lhsT^T @ rhs + bias)

    GELU is computed via the tanh-based Taylor expansion on-chip.
    No intermediate tensor is written to HBM.

    Parameters
    ----------
    lhsT : tensor [K, M]  pre-transposed input activations
    rhs  : tensor [K, N]  weight matrix
    bias : tensor [1, N]  bias vector (broadcast along M)

    Returns
    -------
    result : tensor [M, N]
    """
    K, M = lhsT.shape
    K_, N = rhs.shape
    assert K == K_
    assert M % TILE_M == 0 and N % TILE_N == 0 and K % TILE_K == 0

    result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

    for m in nl.affine_range(M // TILE_M):
        for n in nl.affine_range(N // TILE_N):
            acc = nl.zeros((TILE_M, TILE_N), nl.float32, buffer=nl.psum)

            for k in nl.affine_range(K // TILE_K):
                lhsT_tile = nl.load(
                    lhsT[k * TILE_K:(k + 1) * TILE_K,
                         m * TILE_M:(m + 1) * TILE_M])
                rhs_tile = nl.load(
                    rhs[k * TILE_K:(k + 1) * TILE_K,
                        n * TILE_N:(n + 1) * TILE_N])
                acc += nl.matmul(lhsT_tile, rhs_tile, transpose_x=True)

            # PSUM -> SBUF
            acc_sb = nl.copy(acc, dtype=result.dtype)

            # Bias add (on-chip) — broadcast along partition dim
            bias_tile = nl.load(bias[0:1, n * TILE_N:(n + 1) * TILE_N])
            acc_sb = acc_sb + bias_tile.broadcast_to((TILE_M, TILE_N))

            # GELU via Taylor approximation (on-chip, no HBM round-trip)
            activated = _taylor_gelu(acc_sb)

            # Single HBM write of final result
            nl.store(
                result[m * TILE_M:(m + 1) * TILE_M,
                       n * TILE_N:(n + 1) * TILE_N],
                value=activated)

    return result


# ── High-level wrapper matching baseline API ─────────────────────────

def mlp_fused(X, W, bias, mode="pwl"):
    """Fused MLP forward pass: Z = GELU(X @ W + bias)

    Parameters
    ----------
    X    : ndarray [M, K]  input activations
    W    : ndarray [K, N]  weight matrix
    bias : ndarray [N]     bias vector
    mode : str             "pwl" or "taylor"

    Returns
    -------
    Z : ndarray [M, N]
    """
    lhsT = X.T.copy()
    bias_2d = bias.reshape(1, -1)
    if mode == "taylor":
        return fused_mlp_taylor(lhsT, W, bias_2d)
    return fused_mlp_pwl(lhsT, W, bias_2d)


# ── NumPy reference ──────────────────────────────────────────────────

def numpy_mlp(X, W, bias):
    """NumPy golden reference: GELU(X @ W + bias) using exact erf."""
    from scipy.special import erf
    Y = X @ W + bias
    return 0.5 * Y * (1.0 + erf(Y / np.sqrt(2.0)))


# ── Self-test (requires Trainium) ───────────────────────────────────

if __name__ == "__main__":
    M, K, N = 256, 512, 1024
    X = np.random.randn(M, K).astype(np.float32)
    W = np.random.randn(K, N).astype(np.float32)
    bias = np.random.randn(N).astype(np.float32)

    ref = numpy_mlp(X, W, bias)

    for mode in ("pwl", "taylor"):
        out = mlp_fused(X, W, bias, mode=mode)
        max_err = np.max(np.abs(out - ref))
        ok = np.allclose(out, ref, atol=1e-2)
        status = "PASS" if ok else "FAIL"
        print(f"  fused_{mode:6s}  max_abs_err={max_err:.6e}  [{status}]")
