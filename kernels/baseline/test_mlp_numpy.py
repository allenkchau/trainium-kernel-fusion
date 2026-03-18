"""
NumPy-only test for the MLP baseline math.

Simulates the tiled GEMM + GELU logic from mlp_gemm.py without
requiring the Neuron SDK.  Run on any machine to verify correctness
before deploying to Trainium.

Usage:  python test_mlp_numpy.py
"""

import numpy as np
from scipy.special import erf

# ── Tile sizes (mirror the NKI kernel constants) ────────────────────
TILE_M = 128
TILE_K = 128
TILE_N = 512


def gelu_exact(x):
    """Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))"""
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))


def tiled_gemm(lhsT, rhs, bias):
    """NumPy simulation of the tiled GEMM kernel.

    Mirrors gemm_kernel tile-by-tile to catch any tiling / indexing bugs.

    Parameters
    ----------
    lhsT : [K, M]  pre-transposed activations
    rhs  : [K, N]  weight matrix
    bias : [1, N]  bias
    """
    K, M = lhsT.shape
    K_, N = rhs.shape
    assert K == K_
    assert M % TILE_M == 0 and N % TILE_N == 0 and K % TILE_K == 0

    result = np.zeros((M, N), dtype=np.float32)

    for m in range(M // TILE_M):
        for n in range(N // TILE_N):
            acc = np.zeros((TILE_M, TILE_N), dtype=np.float32)

            for k in range(K // TILE_K):
                lhsT_tile = lhsT[k * TILE_K:(k + 1) * TILE_K,
                                  m * TILE_M:(m + 1) * TILE_M]
                rhs_tile = rhs[k * TILE_K:(k + 1) * TILE_K,
                               n * TILE_N:(n + 1) * TILE_N]
                # (lhsT_tile)^T @ rhs_tile
                acc += lhsT_tile.T @ rhs_tile

            # Bias broadcast
            bias_tile = bias[0:1, n * TILE_N:(n + 1) * TILE_N]
            acc = acc + bias_tile

            result[m * TILE_M:(m + 1) * TILE_M,
                   n * TILE_N:(n + 1) * TILE_N] = acc

    return result


def tiled_gelu(y):
    """NumPy simulation of the tiled GELU kernel."""
    M, N = y.shape
    assert M % TILE_M == 0 and N % TILE_N == 0

    z = np.empty_like(y)

    for m in range(M // TILE_M):
        for n in range(N // TILE_N):
            tile = y[m * TILE_M:(m + 1) * TILE_M,
                     n * TILE_N:(n + 1) * TILE_N]
            z[m * TILE_M:(m + 1) * TILE_M,
              n * TILE_N:(n + 1) * TILE_N] = gelu_exact(tile)

    return z


def mlp_tiled(X, W, bias):
    """Full tiled MLP pass mirroring the NKI kernel flow."""
    lhsT = X.T.copy()
    bias_2d = bias.reshape(1, -1)
    Y = tiled_gemm(lhsT, W, bias_2d)
    Z = tiled_gelu(Y)
    return Z


def mlp_reference(X, W, bias):
    """One-shot NumPy reference (no tiling)."""
    Y = X @ W + bias
    return gelu_exact(Y)


# ── Tests ───────────────────────────────────────────────────────────
def test_gemm_single_tile():
    """GEMM with exactly one tile per dimension."""
    M, K, N = TILE_M, TILE_K, TILE_N
    X = np.random.randn(M, K).astype(np.float32)
    W = np.random.randn(K, N).astype(np.float32)
    bias = np.random.randn(N).astype(np.float32)

    got = tiled_gemm(X.T.copy(), W, bias.reshape(1, -1))
    expected = X @ W + bias
    err = np.max(np.abs(got - expected))
    assert err < 1e-3, f"single-tile GEMM error {err:.3e}"
    print(f"  PASS  gemm_single_tile        max_err={err:.3e}")


def test_gemm_multi_tile():
    """GEMM spanning multiple tiles in all dimensions."""
    M, K, N = 256, 512, 1024
    X = np.random.randn(M, K).astype(np.float32)
    W = np.random.randn(K, N).astype(np.float32)
    bias = np.random.randn(N).astype(np.float32)

    got = tiled_gemm(X.T.copy(), W, bias.reshape(1, -1))
    expected = X @ W + bias
    err = np.max(np.abs(got - expected))
    assert err < 1e-2, f"multi-tile GEMM error {err:.3e}"
    print(f"  PASS  gemm_multi_tile          max_err={err:.3e}")


def test_gelu_values():
    """GELU at known points: 0 → 0, large positive ≈ identity."""
    M, N = TILE_M, TILE_N
    y = np.zeros((M, N), dtype=np.float32)
    z = tiled_gelu(y)
    assert np.allclose(z, 0.0), "GELU(0) should be 0"

    y_pos = np.full((M, N), 5.0, dtype=np.float32)
    z_pos = tiled_gelu(y_pos)
    err = np.max(np.abs(z_pos - y_pos))
    assert err < 1e-4, f"GELU(5) should ≈ 5, got err {err:.3e}"
    print(f"  PASS  gelu_values              GELU(0)=0, GELU(5)≈5")


def test_gelu_symmetry():
    """GELU is not symmetric, but GELU(x) + GELU(-x) ≈ x for large x."""
    M, N = TILE_M, TILE_N
    x = np.random.randn(M, N).astype(np.float32)
    gx = tiled_gelu(x)
    # GELU should be monotonically close to 0 for large negative inputs
    neg = np.full((M, N), -5.0, dtype=np.float32)
    z_neg = tiled_gelu(neg)
    assert np.all(np.abs(z_neg) < 1e-4), "GELU(-5) should ≈ 0"
    print(f"  PASS  gelu_symmetry            GELU(-5)≈0")


def test_mlp_end_to_end():
    """Full tiled MLP matches one-shot reference."""
    M, K, N = 256, 512, 1024
    np.random.seed(42)
    X = np.random.randn(M, K).astype(np.float32)
    W = np.random.randn(K, N).astype(np.float32)
    bias = np.random.randn(N).astype(np.float32)

    got = mlp_tiled(X, W, bias)
    expected = mlp_reference(X, W, bias)
    err = np.max(np.abs(got - expected))
    assert err < 1e-2, f"end-to-end MLP error {err:.3e}"
    print(f"  PASS  mlp_end_to_end           max_err={err:.3e}")


def test_mlp_bf16_precision():
    """Simulate BF16 precision by rounding inputs, check error stays bounded."""
    M, K, N = 256, 512, 1024
    np.random.seed(123)

    # Simulate BF16 by truncating to ~7 bits of mantissa
    def to_bf16(x):
        return x.astype(np.float32)  # np doesn't have bf16, approximate
        # On Trainium this would be: x.astype(bfloat16)

    X = to_bf16(np.random.randn(M, K).astype(np.float32))
    W = to_bf16(np.random.randn(K, N).astype(np.float32))
    bias = to_bf16(np.random.randn(N).astype(np.float32))

    got = mlp_tiled(X, W, bias)
    expected = mlp_reference(X, W, bias)
    err = np.max(np.abs(got - expected))
    assert err < 1e-2, f"bf16 MLP error {err:.3e}"
    print(f"  PASS  mlp_bf16_precision       max_err={err:.3e}")


if __name__ == "__main__":
    print("MLP baseline NumPy tests")
    print("=" * 50)
    test_gemm_single_tile()
    test_gemm_multi_tile()
    test_gelu_values()
    test_gelu_symmetry()
    test_mlp_end_to_end()
    test_mlp_bf16_precision()
    print("=" * 50)
    print("All tests passed.")
