"""
Fused GEMM + Softmax + GEMM attention kernel for AWS Trainium/Trainium2.

Computes self-attention: Output = softmax(Q^T K) @ V^T

Unlike the baseline which writes/reads intermediates through HBM between
each phase, this kernel fuses all three phases under a single outer loop
over Q tiles.  The QK scores stay in PSUM/SBUF, softmax is applied
immediately, and the P@V accumulation follows — no HBM intermediate is
ever allocated.

Additionally applies the FlashAttention-v2 "delayed division" trick:
the softmax normalization (divide by row sum) is postponed until after
the P@V matmul, reducing total FLOPS.

Input layout:  (d_head, seqlen)  — d_head maps to partition dimension
Output layout: (seqlen, d_head)
"""

import numpy as np
from neuronxcc import nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl


def numpy_attention(q, k, v):
    """NumPy reference.  q, k, v: (d_head, seqlen) → (seqlen, d_head)."""
    qk = np.matmul(q.T, k)
    m = np.max(qk, axis=1, keepdims=True)
    e = np.exp(qk - m)
    scores = e / np.sum(e, axis=1, keepdims=True)
    return np.matmul(scores, v.T)


@nki.jit
def fused_attention(q_ref, k_ref, v_ref):
    """
    Fused attention kernel — GEMM, softmax, and GEMM in one pass.

    All intermediate data (QK scores, softmax weights) remain on-chip
    in SBUF/PSUM.  Only the final output is written to HBM.

    Constraints:
      - d_head == 128  (PMAX, maps to partition dimension)
      - seqlen >= 512 and divisible by 512
      - q, k, v must have identical shapes and dtypes
    """
    d_head, seqlen = q_ref.shape

    PMAX = nl.tile_size.pmax                       # 128
    FMAX_S = nl.tile_size.gemm_stationary_fmax     # 128
    FMAX_M = nl.tile_size.gemm_moving_fmax         # 512

    assert d_head == PMAX
    assert q_ref.shape == k_ref.shape == v_ref.shape
    assert seqlen % FMAX_M == 0

    n_tiles_q = seqlen // FMAX_S
    n_tiles_kv = seqlen // FMAX_M

    kernel_out = nl.ndarray(
        (seqlen, d_head), dtype=q_ref.dtype, buffer=nl.shared_hbm)

    # ------------------------------------------------------------------
    # Load full Q, K, V into SBUF once (d_head=128 fits in partition dim)
    # ------------------------------------------------------------------
    q_sbuf = nl.load(q_ref)
    k_sbuf = nl.load(k_ref)
    v_sbuf = nl.load(v_ref)

    # ------------------------------------------------------------------
    # Pre-transpose V tiles once:
    #   V: (d_head=P128, seqlen_chunk=128) → V^T: (seqlen_chunk=P128, d_head=128)
    # These are reused for every Q tile.
    # ------------------------------------------------------------------
    v_t = nl.ndarray(
        (nl.par_dim(PMAX), seqlen // PMAX, PMAX),
        dtype=nl.float32, buffer=nl.sbuf)
    for i_kv in nl.affine_range(seqlen // PMAX):
        v_psum_t = nisa.nc_transpose(
            v_sbuf[:, nl.ds(i_kv * PMAX, PMAX)])
        v_t[:, i_kv, :] = nisa.tensor_copy(v_psum_t, dtype=nl.float32)

    # ==================================================================
    # Fused loop: for each Q tile, compute QK → softmax → PV in SBUF
    # ==================================================================
    for i_q in nl.affine_range(n_tiles_q):

        # ==============================================================
        # Step 1: GEMM — QK scores (stay in PSUM, then move to SBUF)
        #
        # Score tile shape: (PMAX=128 partition, seqlen free)
        # Each nc_matmul produces a (PMAX, FMAX_M) chunk.
        # ==============================================================
        qk_sbuf = nl.ndarray(
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
            qk_sbuf[:, nl.ds(i_kv * FMAX_M, FMAX_M)] = \
                nisa.tensor_copy(qk_psum, dtype=nl.float32)

        # ==============================================================
        # Step 2: Numerically-stable softmax (entirely on-chip)
        #
        # Row max → subtract → exp → row sum → (delay division)
        # ==============================================================

        # --- Row-wise max ---
        partial_max = nl.ndarray(
            (nl.par_dim(PMAX), n_tiles_kv),
            dtype=nl.float32, buffer=nl.sbuf)
        for j in nl.affine_range(n_tiles_kv):
            partial_max[:, j] = nisa.tensor_reduce(
                op=nl.max,
                data=qk_sbuf[:, nl.ds(j * FMAX_M, FMAX_M)],
                axis=1)
        neg_max = nisa.tensor_reduce(
            op=nl.max, data=partial_max, axis=1, negate=True)

        # --- exp(scores - max) and partial row sums ---
        exp_tile = nl.ndarray(
            (nl.par_dim(PMAX), seqlen), dtype=nl.float32, buffer=nl.sbuf)
        sum_chunks = nl.ndarray(
            (nl.par_dim(PMAX), n_tiles_kv),
            dtype=nl.float32, buffer=nl.sbuf)
        for j in nl.affine_range(n_tiles_kv):
            exp_tile[:, nl.ds(j * FMAX_M, FMAX_M)] = nisa.activation(
                op=nl.exp,
                data=qk_sbuf[:, nl.ds(j * FMAX_M, FMAX_M)],
                bias=neg_max, scale=1.0)
            sum_chunks[:, j] = nisa.tensor_reduce(
                op=nl.add,
                data=exp_tile[:, nl.ds(j * FMAX_M, FMAX_M)],
                axis=1)

        row_sum = nisa.tensor_reduce(op=nl.add, data=sum_chunks, axis=1)
        inv_sum = nisa.reciprocal(data=row_sum)

        # ==============================================================
        # Step 3: Transpose softmax(exp) for P @ V matmul
        #
        # exp_tile layout:   (seqlen_q_tile=P128, seqlen_kv)
        # We need:           (seqlen_kv_chunk=P128, seqlen_q_tile=128)
        # so that the contraction axis (seqlen_kv) is in partition dim.
        #
        # Division by row_sum is DELAYED until after the matmul
        # (FlashAttention-v2 trick: saves seqlen worth of divides).
        # ==============================================================
        scores_t = nl.ndarray(
            (nl.par_dim(PMAX), seqlen // PMAX, PMAX),
            dtype=nl.float32, buffer=nl.sbuf)
        for i_kv in nl.affine_range(seqlen // PMAX):
            scores_psum_t = nisa.nc_transpose(
                exp_tile[:, nl.ds(i_kv * PMAX, PMAX)])
            scores_t[:, i_kv, :] = nisa.tensor_copy(
                scores_psum_t, dtype=nl.float32)

        # ==============================================================
        # Step 4: GEMM — accumulate P @ V^T  (on-chip)
        #
        # stationary = scores_t  (P128 = seqlen_kv_chunk, F128 = seqlen_q)
        # moving     = v_t       (P128 = seqlen_kv_chunk, F128 = d_head)
        # result     = (P128 = d_head???) — actually result partition dim
        #              corresponds to stationary free = seqlen_q_tile,
        #              and result free = moving free = d_head.
        # ==============================================================
        attn_psum = nl.zeros(
            (PMAX, PMAX), dtype=nl.float32, buffer=nl.psum)
        for i_kv in nl.affine_range(seqlen // PMAX):
            attn_psum += nisa.nc_matmul(
                stationary=scores_t[:, i_kv, :],
                moving=v_t[:, i_kv, :])

        # ==============================================================
        # Step 5: Delayed softmax division — multiply by 1/row_sum
        #
        # attn_psum shape: (PMAX, PMAX)  = (seqlen_q_tile, d_head)
        # inv_sum shape:   (PMAX, 1)     = (seqlen_q_tile, 1)
        # ==============================================================
        attn_out = nl.ndarray(
            (nl.par_dim(PMAX), PMAX), dtype=nl.float32, buffer=nl.sbuf)
        attn_out[...] = nisa.tensor_scalar(
            data=attn_psum,
            op0=nl.multiply, operand0=inv_sum,
            engine=nisa.vector_engine)

        # ==============================================================
        # Store — only HBM write in the entire fused loop
        # ==============================================================
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
    actual = fused_attention(q, k, v)

    if np.allclose(actual, expected, atol=1e-2):
        print("PASS: fused_attention matches NumPy reference")
    else:
        max_err = np.max(np.abs(actual - expected))
        print(f"FAIL: max abs error = {max_err}")
