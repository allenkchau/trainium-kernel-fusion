"""
Baseline (non-fused) NKI attention kernel for AWS Trainium/Trainium2.

Computes self-attention: Output = softmax(Q^T K) @ V^T

Three phases with explicit HBM intermediates demonstrate the memory
traffic overhead that kernel fusion eliminates:

  Phase 1: GEMM    — S = Q^T K       → write to HBM
  Phase 2: Softmax — P = softmax(S)  → write to HBM (overwrites S)
  Phase 3: GEMM    — O = P @ V^T     → write to HBM

Input layout:  (d_head, seqlen) — d_head maps to partition dimension
Output layout: (seqlen, d_head)
"""

import numpy as np
from neuronxcc import nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl


def numpy_attention(q, k, v):
    """
    NumPy reference attention.

    Args:
        q, k, v: shape (d_head, seqlen)
    Returns:
        (seqlen, d_head) attention output
    """
    qk = np.matmul(q.T, k)
    m = np.max(qk, axis=1, keepdims=True)
    e = np.exp(qk - m)
    scores = e / np.sum(e, axis=1, keepdims=True)
    return np.matmul(scores, v.T)


@nki.jit
def baseline_attention(q_ref, k_ref, v_ref):
    """
    Non-fused baseline attention with tiling.

    The three computation phases are separated by HBM round-trips,
    creating the memory traffic bottleneck that the fused
    GEMM + Softmax kernel eliminates.

    Constraints:
      - d_head == 128  (PMAX, maps to partition dimension)
      - seqlen >= 512 and divisible by 512
      - q, k, v must have identical shapes and dtypes
    """
    d_head, seqlen = q_ref.shape

    PMAX = nl.tile_size.pmax
    FMAX_S = nl.tile_size.gemm_stationary_fmax   # 128
    FMAX_M = nl.tile_size.gemm_moving_fmax       # 512

    assert d_head == PMAX
    assert q_ref.shape == k_ref.shape == v_ref.shape
    assert seqlen % FMAX_M == 0

    n_tiles_q = seqlen // FMAX_S
    n_tiles_kv = seqlen // FMAX_M

    kernel_out = nl.ndarray(
        (seqlen, d_head), dtype=q_ref.dtype, buffer=nl.shared_hbm)

    # Load full Q, K, V into SBUF (fits because d_head=128 is partition)
    q_sbuf = nl.load(q_ref)
    k_sbuf = nl.load(k_ref)
    v_sbuf = nl.load(v_ref)

    ####################################################################
    # Phase 1: GEMM — S = Q^T K → HBM
    #
    # Tiles Q along seqlen in chunks of FMAX_S (128) for the stationary
    # operand, and K in chunks of FMAX_M (512) for the moving operand.
    # The full score matrix is written to HBM.
    ####################################################################
    score_hbm = nl.ndarray(
        (n_tiles_q, PMAX, seqlen), dtype=nl.float32, buffer=nl.shared_hbm)

    for i_q in nl.affine_range(n_tiles_q):
        qk_buf = nl.ndarray(
            (nl.par_dim(PMAX), seqlen), dtype=nl.float32, buffer=nl.sbuf)

        for i_kv in nl.affine_range(n_tiles_kv):
            qk_psum = nl.ndarray(
                (nl.par_dim(PMAX), FMAX_M),
                dtype=nl.float32, buffer=nl.psum)
            qk_psum[...] = nisa.nc_matmul(
                stationary=q_sbuf[0:PMAX,
                                  nl.ds(i_q * FMAX_S, FMAX_S)],
                moving=k_sbuf[0:PMAX,
                              nl.ds(i_kv * FMAX_M, FMAX_M)])

            # PSUM → SBUF
            qk_buf[:, nl.ds(i_kv * FMAX_M, FMAX_M)] = \
                nisa.tensor_copy(qk_psum, dtype=nl.float32)

        # SBUF → HBM  (the non-fused bottleneck)
        nl.store(score_hbm[i_q], qk_buf)

    ####################################################################
    # Phase 2: Softmax — P = softmax(S) → HBM
    #
    # Reads each (PMAX, seqlen) tile of scores from HBM, applies
    # numerically-stable softmax per row, and writes the attention
    # weights back to HBM (reusing the score buffer).
    ####################################################################
    for i_q in nl.affine_range(n_tiles_q):
        # HBM → SBUF
        s_tile = nl.ndarray(
            (nl.par_dim(PMAX), seqlen), dtype=nl.float32, buffer=nl.sbuf)
        s_tile[...] = nl.load(score_hbm[i_q])

        # --- Row-wise max ---
        partial_max = nl.ndarray(
            (nl.par_dim(PMAX), n_tiles_kv),
            dtype=nl.float32, buffer=nl.sbuf)
        for j in nl.affine_range(n_tiles_kv):
            partial_max[:, j] = nisa.tensor_reduce(
                op=nl.max,
                data=s_tile[:, nl.ds(j * FMAX_M, FMAX_M)],
                axis=1)
        neg_max = nisa.tensor_reduce(
            op=nl.max, data=partial_max, axis=1, negate=True)

        # --- exp(s - max) and partial sums ---
        exp_tile = nl.ndarray(
            (nl.par_dim(PMAX), seqlen), dtype=nl.float32, buffer=nl.sbuf)
        sum_chunks = nl.ndarray(
            (nl.par_dim(PMAX), n_tiles_kv),
            dtype=nl.float32, buffer=nl.sbuf)
        for j in nl.affine_range(n_tiles_kv):
            exp_tile[:, nl.ds(j * FMAX_M, FMAX_M)] = nisa.activation(
                op=nl.exp,
                data=s_tile[:, nl.ds(j * FMAX_M, FMAX_M)],
                bias=neg_max, scale=1.0)
            sum_chunks[:, j] = nisa.tensor_reduce(
                op=nl.add,
                data=exp_tile[:, nl.ds(j * FMAX_M, FMAX_M)],
                axis=1)

        # --- Normalize ---
        row_sum = nisa.tensor_reduce(op=nl.add, data=sum_chunks, axis=1)
        inv_sum = nisa.reciprocal(data=row_sum)

        weight_tile = nl.ndarray(
            (nl.par_dim(PMAX), seqlen), dtype=nl.float32, buffer=nl.sbuf)
        for j in nl.affine_range(n_tiles_kv):
            weight_tile[:, nl.ds(j * FMAX_M, FMAX_M)] = nisa.tensor_scalar(
                data=exp_tile[:, nl.ds(j * FMAX_M, FMAX_M)],
                op0=nl.multiply, operand0=inv_sum,
                engine=nisa.vector_engine)

        # SBUF → HBM  (overwrite scores with attention weights)
        nl.store(score_hbm[i_q], weight_tile)

    ####################################################################
    # Phase 3: GEMM — O = P @ V^T → HBM
    #
    # Reads attention weights P from HBM. Transposes P and V into the
    # layout required by nc_matmul (contraction axis in partition dim).
    # Accumulates over seqlen_kv tiles to produce (seqlen_q, d_head).
    ####################################################################

    # Pre-transpose V tiles: (d_head=128, seqlen) → (seqlen_tile=128, d_head=128)
    v_t = nl.ndarray(
        (nl.par_dim(PMAX), seqlen // PMAX, PMAX),
        dtype=nl.float32, buffer=nl.sbuf)
    for i_kv in nl.affine_range(seqlen // PMAX):
        v_psum_t = nisa.nc_transpose(
            v_sbuf[:, nl.ds(i_kv * PMAX, PMAX)])
        v_t[:, i_kv, :] = nisa.tensor_copy(v_psum_t, dtype=nl.float32)

    for i_q in nl.affine_range(n_tiles_q):
        # Read attention weights from HBM
        p_tile = nl.ndarray(
            (nl.par_dim(PMAX), seqlen), dtype=nl.float32, buffer=nl.sbuf)
        p_tile[...] = nl.load(score_hbm[i_q])

        # Transpose weight tiles: (seqlen_q_tile=128, seqlen_kv_chunk=128)
        # → (seqlen_kv_chunk=128, seqlen_q_tile=128)
        p_t = nl.ndarray(
            (nl.par_dim(PMAX), seqlen // PMAX, PMAX),
            dtype=nl.float32, buffer=nl.sbuf)
        for i_kv in nl.affine_range(seqlen // PMAX):
            p_psum_t = nisa.nc_transpose(
                p_tile[:, nl.ds(i_kv * PMAX, PMAX)])
            p_t[:, i_kv, :] = nisa.tensor_copy(
                p_psum_t, dtype=nl.float32)

        # Accumulate: P^T @ V^T over seqlen_kv tiles
        # stationary(par=seqlen_kv, free=seqlen_q) →  result partition = seqlen_q
        # moving(par=seqlen_kv, free=d_head)       →  result free = d_head
        attn_psum = nl.zeros(
            (PMAX, PMAX), dtype=nl.float32, buffer=nl.psum)
        for i_kv in nl.affine_range(seqlen // PMAX):
            attn_psum += nisa.nc_matmul(
                stationary=p_t[:, i_kv, :],
                moving=v_t[:, i_kv, :])

        attn_out = nl.ndarray(
            (nl.par_dim(PMAX), PMAX), dtype=nl.float32, buffer=nl.sbuf)
        attn_out[...] = nisa.tensor_copy(attn_psum, dtype=nl.float32)

        nl.store(
            dst=kernel_out[nl.ds(i_q * PMAX, PMAX), :],
            value=attn_out)

    return kernel_out


# ---------------------------------------------------------------------------
# Testing (requires Trainium instance)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    d_head = 128
    seqlen = 4096

    q = ((np.random.random_sample([d_head, seqlen]) - 0.5) * 2).astype(
        np.float32)
    k = ((np.random.random_sample([d_head, seqlen]) - 0.5) * 2).astype(
        np.float32)
    v = ((np.random.random_sample([d_head, seqlen]) - 0.5) * 2).astype(
        np.float32)

    expected = numpy_attention(q, k, v)
    actual = baseline_attention(q, k, v)

    if np.allclose(actual, expected, atol=1e-2):
        print("PASS: baseline_attention matches NumPy reference")
    else:
        max_err = np.max(np.abs(actual - expected))
        print(f"FAIL: max abs error = {max_err}")
