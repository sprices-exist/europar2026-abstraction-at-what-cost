import math
import torch
import cuda.tile as ct
import triton
import triton.language as tl


HEAD_DIM = 128
BLOCK_SIZE = 16


def pytorch_paged_attention(q, k_cache, v_cache, block_table, scale):
    """Reference implementation used for correctness and timing."""
    B, H, D = q.shape
    out = torch.empty_like(q)

    for b in range(B):
        active_block_indices = block_table[b]
        k_blocks = k_cache[active_block_indices]
        v_blocks = v_cache[active_block_indices]

        k_seq = k_blocks.permute(1, 0, 3, 2).reshape(H, -1, D)
        v_seq = v_blocks.permute(1, 0, 3, 2).reshape(H, -1, D)

        q_head = q[b].unsqueeze(1)
        scores = torch.matmul(q_head, k_seq.transpose(-2, -1)) * scale
        probs = torch.softmax(scores, dim=-1)
        attn_out = torch.matmul(probs, v_seq)
        out[b] = attn_out.squeeze(1)

    return out


@ct.kernel
def paged_attention_kernel_old(
    Q: ct.Array,
    K_Cache: ct.Array,
    V_Cache: ct.Array,
    BlockTable: ct.Array,
    Out: ct.Array,
    Idx_Dim: ct.Array,
    Idx_Block: ct.Array,
    sm_scale: float,
    neg_inf: float,
    B_SIZE: ct.Constant[int],
    H_DIM: ct.Constant[int],
    MAX_LOOP: ct.Constant[int],
):
    """Original cuTile implementation kept for apples-to-apples comparison."""
    inv_log_2 = 1.44269504
    qk_scale = sm_scale * inv_log_2
    batch_id = ct.bid(0)
    head_id = ct.bid(1)

    q_tile_3d = ct.load(Q, index=(batch_id, head_id, 0), shape=(1, 1, H_DIM))
    q_tile = ct.reshape(q_tile_3d, (1, H_DIM))

    idx_head_tile = ct.zeros((1, 1), dtype=ct.int32) + head_id
    idx_dim_tile = ct.load(Idx_Dim, index=(0, 0), shape=(H_DIM, 1))
    idx_blk_tile = ct.load(Idx_Block, index=(0, 0), shape=(1, B_SIZE))

    acc_o = ct.zeros((1, H_DIM), dtype=ct.float32)
    m_i = ct.full((1, 1), neg_inf, dtype=ct.float32)
    l_i = ct.zeros((1, 1), dtype=ct.float32)

    for i in range(MAX_LOOP):
        phys_block_tile = ct.load(BlockTable, index=(batch_id, i), shape=(1, 1))
        k_tile = ct.gather(K_Cache, (phys_block_tile, idx_head_tile, idx_dim_tile, idx_blk_tile))
        k_tile_trans = ct.transpose(k_tile)
        qk_product = q_tile * k_tile_trans
        scores = ct.sum(qk_product, axis=1)
        scores = ct.reshape(scores, (1, B_SIZE))
        scores = scores * qk_scale

        row_max = ct.max(scores, axis=1, keepdims=True)
        m_new = ct.maximum(m_i, row_max)
        p_tile = ct.exp2(scores - m_new)
        alpha = ct.exp2(m_i - m_new)

        row_sum = ct.sum(p_tile, axis=1, keepdims=True)
        l_i = (l_i * alpha) + row_sum

        acc_o = acc_o * alpha
        v_tile = ct.gather(V_Cache, (phys_block_tile, idx_head_tile, idx_dim_tile, idx_blk_tile))
        pv_sum = ct.sum(v_tile * p_tile, axis=1)
        pv_reshaped = ct.reshape(pv_sum, (1, H_DIM))
        acc_o = acc_o + pv_reshaped
        m_i = m_new

    final_output = acc_o / (l_i + 1e-6)
    final_output_3d = ct.reshape(final_output, (1, 1, H_DIM)).astype(Out.dtype)
    ct.store(Out, index=(batch_id, head_id, 0), tile=final_output_3d)


@ct.kernel
def paged_attention_kernel_tuned(
    Q: ct.Array,
    K_Cache: ct.Array,
    V_Cache: ct.Array,
    BlockTable: ct.Array,
    Out: ct.Array,
    sm_scale: float,
    neg_inf: float,
    B_SIZE: ct.Constant[int],
    H_DIM: ct.Constant[int],
    MAX_LOOP: ct.Constant[int],
):
    """
    A lighter cuTile variant:
    - uses compile-time index tiles instead of loading index tensors
    - disables gather bounds checks when indices are known-valid
    - adds gather latency hints so the compiler can prioritize the DRAM-heavy path
    """
    inv_log_2 = 1.44269504
    qk_scale = sm_scale * inv_log_2
    batch_id = ct.bid(0)
    head_id = ct.bid(1)

    q_tile_3d = ct.load(Q, index=(batch_id, head_id, 0), shape=(1, 1, H_DIM), latency=4)
    q_tile = ct.reshape(q_tile_3d, (1, H_DIM))

    idx_dim_tile = ct.reshape(ct.arange(H_DIM, dtype=ct.int32), (H_DIM, 1))
    idx_blk_tile = ct.reshape(ct.arange(B_SIZE, dtype=ct.int32), (1, B_SIZE))

    acc_o = ct.zeros((1, H_DIM), dtype=ct.float32)
    m_i = ct.full((1, 1), neg_inf, dtype=ct.float32)
    l_i = ct.zeros((1, 1), dtype=ct.float32)

    for i in range(MAX_LOOP):
        phys_block = ct.load(BlockTable, index=(batch_id, i), shape=())
        k_tile = ct.gather(
            K_Cache,
            (phys_block, head_id, idx_dim_tile, idx_blk_tile),
            check_bounds=False,
            latency=10,
        )
        k_tile_trans = ct.transpose(k_tile)
        scores = ct.sum(q_tile * k_tile_trans, axis=1)
        scores = ct.reshape(scores, (1, B_SIZE))
        scores = scores * qk_scale

        row_max = ct.max(scores, axis=1, keepdims=True)
        m_new = ct.maximum(m_i, row_max)
        p_tile = ct.exp2(scores - m_new)
        alpha = ct.exp2(m_i - m_new)

        row_sum = ct.sum(p_tile, axis=1, keepdims=True)
        l_i = (l_i * alpha) + row_sum

        acc_o = acc_o * alpha
        v_tile = ct.gather(
            V_Cache,
            (phys_block, head_id, idx_dim_tile, idx_blk_tile),
            check_bounds=False,
            latency=10,
        )
        pv_sum = ct.sum(v_tile * p_tile, axis=1)
        acc_o = acc_o + ct.reshape(pv_sum, (1, H_DIM))
        m_i = m_new

    final_output = acc_o / (l_i + 1e-6)
    final_output_3d = ct.reshape(final_output, (1, 1, H_DIM)).astype(Out.dtype)
    ct.store(Out, index=(batch_id, head_id, 0), tile=final_output_3d)


@ct.kernel(
    occupancy=ct.ByTarget(sm_120=8, default=4),
    opt_level=3,
)
def paged_attention_kernel_aggressive(
    Q: ct.Array,
    K_Cache: ct.Array,
    V_Cache: ct.Array,
    BlockTable: ct.Array,
    Out: ct.Array,
    sm_scale: float,
    neg_inf: float,
    B_SIZE: ct.Constant[int],
    H_DIM: ct.Constant[int],
    MAX_LOOP: ct.Constant[int],
):
    """
    More aggressive cuTile variant using the same math as the tuned kernel,
    but with explicit kernel configuration hints for Blackwell.
    """
    inv_log_2 = 1.44269504
    qk_scale = sm_scale * inv_log_2
    batch_id = ct.bid(0)
    head_id = ct.bid(1)

    q_tile_3d = ct.load(Q, index=(batch_id, head_id, 0), shape=(1, 1, H_DIM), latency=2)
    q_tile = ct.reshape(q_tile_3d, (1, H_DIM))

    idx_dim_tile = ct.reshape(ct.arange(H_DIM, dtype=ct.int32), (H_DIM, 1))
    idx_blk_tile = ct.reshape(ct.arange(B_SIZE, dtype=ct.int32), (1, B_SIZE))

    acc_o = ct.zeros((1, H_DIM), dtype=ct.float32)
    m_i = ct.full((1, 1), neg_inf, dtype=ct.float32)
    l_i = ct.zeros((1, 1), dtype=ct.float32)

    for i in range(MAX_LOOP):
        phys_block = ct.load(BlockTable, index=(batch_id, i), shape=(), latency=2)
        k_tile = ct.gather(
            K_Cache,
            (phys_block, head_id, idx_dim_tile, idx_blk_tile),
            check_bounds=False,
            latency=10,
        )
        scores = ct.reshape(ct.sum(q_tile * ct.transpose(k_tile), axis=1), (1, B_SIZE))
        scores = scores * qk_scale

        row_max = ct.max(scores, axis=1, keepdims=True)
        m_new = ct.maximum(m_i, row_max)
        p_tile = ct.exp2(scores - m_new)
        alpha = ct.exp2(m_i - m_new)

        l_i = (l_i * alpha) + ct.sum(p_tile, axis=1, keepdims=True)
        acc_o = acc_o * alpha

        v_tile = ct.gather(
            V_Cache,
            (phys_block, head_id, idx_dim_tile, idx_blk_tile),
            check_bounds=False,
            latency=10,
        )
        acc_o = acc_o + ct.reshape(ct.sum(v_tile * p_tile, axis=1), (1, H_DIM))
        m_i = m_new

    final_output = acc_o / (l_i + 1e-6)
    final_output_3d = ct.reshape(final_output, (1, 1, H_DIM)).astype(Out.dtype)
    ct.store(Out, index=(batch_id, head_id, 0), tile=final_output_3d, latency=2)


@triton.jit
def triton_paged_attn_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    block_table_ptr,
    out_ptr,
    stride_qb,
    stride_qh,
    stride_qd,
    stride_kp,
    stride_kh,
    stride_kd,
    stride_kb,
    stride_btb,
    stride_btmax,
    stride_ob,
    stride_oh,
    stride_od,
    sm_scale,
    MAX_BLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)
    dim_offsets = tl.arange(0, HEAD_DIM)
    blk_offsets = tl.arange(0, BLOCK_SIZE)

    q_ptrs = q_ptr + batch_id * stride_qb + head_id * stride_qh + dim_offsets * stride_qd
    q = tl.load(q_ptrs)

    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    for block_idx in range(MAX_BLOCKS):
        bt_ptr = block_table_ptr + batch_id * stride_btb + block_idx * stride_btmax
        phys_block = tl.load(bt_ptr)
        kv_base_offset = phys_block * stride_kp + head_id * stride_kh

        k_ptrs = k_cache_ptr + kv_base_offset + dim_offsets[:, None] * stride_kd + blk_offsets[None, :] * stride_kb
        v_ptrs = v_cache_ptr + kv_base_offset + dim_offsets[:, None] * stride_kd + blk_offsets[None, :] * stride_kb

        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)

        qk = tl.sum(q[:, None] * k, axis=0) * sm_scale
        m_ij = tl.max(qk)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new)

        acc = acc * alpha + tl.sum(v * p[None, :], axis=1)
        l_i = l_i * alpha + tl.sum(p)
        m_i = m_new

    out = acc / (l_i + 1e-6)
    out_ptrs = out_ptr + batch_id * stride_ob + head_id * stride_oh + dim_offsets * stride_od
    tl.store(out_ptrs, out)


def benchmark_func(name, func, iters=1000, warmup=10):
    for _ in range(warmup):
        func()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        func()
    end.record()
    torch.cuda.synchronize()

    avg_ms = start.elapsed_time(end) / iters
    print(f"{name:<28}: {avg_ms * 1000:.2f} us")
    return avg_ms * 1000


def run_once_and_sync(name, func):
    """
    Force a sync after a single launch so asynchronous CUDA failures
    are attributed to the kernel that triggered them.
    """
    try:
        func()
        torch.cuda.synchronize()
    except Exception as exc:
        raise RuntimeError(f"{name} failed during launch or execution") from exc


def run_benchmarks(context_len=2048, dtype_name="bf16"):
    B, H = 16, 16
    D = HEAD_DIM
    max_blocks = context_len // BLOCK_SIZE
    pool_size = max(5000, max_blocks * B)
    scale = 1.0 / math.sqrt(D)

    if dtype_name == "bf16":
        dtype = torch.bfloat16
        tol = 0.02
    elif dtype_name == "fp32":
        dtype = torch.float32
        tol = 1e-2
    else:
        raise ValueError("dtype_name must be 'bf16' or 'fp32'")

    print(f"Paged Attention Variants: B={B}, H={H}, D={D}, CTX={context_len}, dtype={dtype_name}")

    t_q = torch.randn(B, H, D, device="cuda", dtype=dtype)
    t_k = torch.randn(pool_size, H, D, BLOCK_SIZE, device="cuda", dtype=dtype)
    t_v = torch.randn(pool_size, H, D, BLOCK_SIZE, device="cuda", dtype=dtype)
    t_bt = torch.randint(0, pool_size, (B, max_blocks), device="cuda", dtype=torch.int32)

    old_out = torch.empty_like(t_q)
    tuned_out = torch.empty_like(t_q)
    aggressive_out = torch.empty_like(t_q)
    tri_out = torch.empty_like(t_q)

    idx_dim = torch.arange(D, dtype=torch.int32, device="cuda").reshape(D, 1)
    idx_blk = torch.arange(BLOCK_SIZE, dtype=torch.int32, device="cuda").reshape(1, BLOCK_SIZE)
    neg_inf = -float("inf")
    ct_stream = torch.cuda.current_stream()
    tri_grid = (B, H)

    def run_old():
        ct.launch(
            ct_stream,
            (B, H, 1),
            paged_attention_kernel_old,
            (t_q, t_k, t_v, t_bt, old_out, idx_dim, idx_blk, scale, neg_inf, BLOCK_SIZE, D, max_blocks),
        )

    def run_tuned():
        ct.launch(
            ct_stream,
            (B, H, 1),
            paged_attention_kernel_tuned,
            (t_q, t_k, t_v, t_bt, tuned_out, scale, neg_inf, BLOCK_SIZE, D, max_blocks),
        )

    def run_aggressive():
        ct.launch(
            ct_stream,
            (B, H, 1),
            paged_attention_kernel_aggressive,
            (t_q, t_k, t_v, t_bt, aggressive_out, scale, neg_inf, BLOCK_SIZE, D, max_blocks),
        )

    def run_triton():
        triton_paged_attn_kernel[tri_grid](
            t_q,
            t_k,
            t_v,
            t_bt,
            tri_out,
            t_q.stride(0),
            t_q.stride(1),
            t_q.stride(2),
            t_k.stride(0),
            t_k.stride(1),
            t_k.stride(2),
            t_k.stride(3),
            t_bt.stride(0),
            t_bt.stride(1),
            tri_out.stride(0),
            tri_out.stride(1),
            tri_out.stride(2),
            scale,
            max_blocks,
            BLOCK_SIZE,
            D,
        )

    print("Verifying correctness...")
    ref_out = pytorch_paged_attention(t_q, t_k, t_v, t_bt, scale)

    run_once_and_sync("cuTile old", run_old)
    run_once_and_sync("cuTile tuned", run_tuned)
    run_once_and_sync("cuTile aggressive", run_aggressive)
    run_once_and_sync("Triton", run_triton)

    diff_old = torch.max(torch.abs(ref_out - old_out)).item()
    diff_tuned = torch.max(torch.abs(ref_out - tuned_out)).item()
    diff_aggressive = torch.max(torch.abs(ref_out - aggressive_out)).item()
    diff_triton = torch.max(torch.abs(ref_out - tri_out)).item()

    print(f"Max Diff (PyTorch vs cuTile old)  : {diff_old:.6f}")
    print(f"Max Diff (PyTorch vs cuTile tuned): {diff_tuned:.6f}")
    print(f"Max Diff (PyTorch vs cuTile aggr) : {diff_aggressive:.6f}")
    print(f"Max Diff (PyTorch vs Triton)      : {diff_triton:.6f}")

    assert diff_old < tol, f"old cuTile kernel incorrect: diff={diff_old}"
    assert diff_tuned < tol, f"tuned cuTile kernel incorrect: diff={diff_tuned}"
    assert diff_aggressive < tol, f"aggressive cuTile kernel incorrect: diff={diff_aggressive}"
    assert diff_triton < tol, f"Triton kernel incorrect: diff={diff_triton}"

    print("\nBenchmarking...")
    pyt_us = benchmark_func("PyTorch (Reference)", lambda: pytorch_paged_attention(t_q, t_k, t_v, t_bt, scale))
    old_us = benchmark_func("cuTile (old)", run_old)
    tuned_us = benchmark_func("cuTile (tuned)", run_tuned)
    aggressive_us = benchmark_func("cuTile (aggressive)", run_aggressive)
    tri_us = benchmark_func("Triton", run_triton)

    print("\nSpeedups")
    print(f"tuned / old cuTile : {old_us / tuned_us:.3f}x")
    print(f"aggr / tuned cuTile: {tuned_us / aggressive_us:.3f}x")
    print(f"aggr / old cuTile  : {old_us / aggressive_us:.3f}x")
    print(f"Triton / tuned      : {tuned_us / tri_us:.3f}x slower")
    print(f"Triton / aggressive : {aggressive_us / tri_us:.3f}x slower")
    print(f"Triton / old        : {old_us / tri_us:.3f}x slower")
    print(f"PyTorch / tuned     : {pyt_us / tuned_us:.3f}x")
    print(f"PyTorch / aggr      : {pyt_us / aggressive_us:.3f}x")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare old/tuned cuTile paged-attention kernels against Triton.")
    parser.add_argument("--context-len", type=int, default=2048, help="Context length in tokens (multiple of 16).")
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16", help="Benchmark datatype.")
    args = parser.parse_args()

    assert args.context_len % BLOCK_SIZE == 0, f"context length must be a multiple of {BLOCK_SIZE}"
    run_benchmarks(context_len=args.context_len, dtype_name=args.dtype)
