import time
import math
import torch
import torch.nn.functional as F
import cupy as cp
import cuda.tile as ct
import triton
import triton.language as tl

# --- Constants ---
HEAD_DIM = 128
BLOCK_SIZE = 16

# ==========================================
# 1. PyTorch Implementation (Reference)
# ==========================================
def pytorch_paged_attention(q, k_cache, v_cache, block_table, scale):
    """
    Pure PyTorch implementation of Paged Attention.
    
    Args:
        q: [B, H, D] - Query tensor
        k_cache: [Num_Blocks, H, D, Block_Size] - Key Cache
        v_cache: [Num_Blocks, H, D, Block_Size] - Value Cache
        block_table: [B, Max_Blocks_Per_Seq] - Indices into cache
        scale: softmax scaling factor
    """
    B, H, D = q.shape
    # Output tensor
    out = torch.empty_like(q)
    
    # Iterate over batch (simulating the paging logic)
    for b in range(B):
        # 1. Get active blocks for this sequence
        # block_table[b] shape: [Max_Blocks]
        active_block_indices = block_table[b]
        
        # 2. Gather the physical blocks
        # k_cache shape: [Pool, H, D, Blk]
        # selected shape: [Max_Blocks, H, D, Blk]
        k_blocks = k_cache[active_block_indices]
        v_blocks = v_cache[active_block_indices]
        
        # 3. Reshape to contiguous sequence for standard attention
        # Permute: [Max_Blocks, H, D, Blk] -> [H, Max_Blocks, Blk, D]
        # Reshape: -> [H, Seq_Len, D] where Seq_Len = Max_Blocks * Blk
        k_seq = k_blocks.permute(1, 0, 3, 2).reshape(H, -1, D)
        v_seq = v_blocks.permute(1, 0, 3, 2).reshape(H, -1, D)
        
        # 4. Standard Attention Calculation
        # q[b]: [H, D] -> [H, 1, D]
        q_head = q[b].unsqueeze(1)
        
        # Scores: [H, 1, D] @ [H, D, Seq_Len] -> [H, 1, Seq_Len]
        scores = torch.matmul(q_head, k_seq.transpose(-2, -1)) * scale
        probs = torch.softmax(scores, dim=-1)
        
        # Output: [H, 1, Seq_Len] @ [H, Seq_Len, D] -> [H, 1, D]
        attn_out = torch.matmul(probs, v_seq)
        
        out[b] = attn_out.squeeze(1)
        
    return out

# ==========================================
# 2. cuTile Kernel
# ==========================================
@ct.kernel
def paged_attention_kernel(
    Q: ct.Array, K_Cache: ct.Array, V_Cache: ct.Array, BlockTable: ct.Array, Out: ct.Array,
    Idx_Dim: ct.Array, Idx_Block: ct.Array,
    sm_scale: float, neg_inf: float,
    B_SIZE: ct.Constant[int], H_DIM: ct.Constant[int], MAX_LOOP: ct.Constant[int]
):
    INV_LOG_2 = 1.44269504
    qk_scale = sm_scale * INV_LOG_2
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
    final_output_3d = ct.reshape(final_output, (1, 1, H_DIM))
    ct.store(Out, index=(batch_id, head_id, 0), tile=final_output_3d)

# ==========================================
# 3. Triton Kernel
# ==========================================
@triton.jit
def triton_paged_attn_kernel(
    q_ptr, k_cache_ptr, v_cache_ptr, block_table_ptr, out_ptr,
    stride_qb, stride_qh, stride_qd,
    stride_kp, stride_kh, stride_kd, stride_kb,
    stride_btb, stride_btmax,
    stride_ob, stride_oh, stride_od,
    sm_scale, MAX_BLOCKS: tl.constexpr, BLOCK_SIZE: tl.constexpr, HEAD_DIM: tl.constexpr
):
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)
    dim_offsets = tl.arange(0, HEAD_DIM)
    blk_offsets = tl.arange(0, BLOCK_SIZE)
    
    q_ptrs = q_ptr + batch_id * stride_qb + head_id * stride_qh + dim_offsets * stride_qd
    q = tl.load(q_ptrs)
    
    m_i = -float('inf')
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

# ==========================================
# 4. Benchmark Runner
# ==========================================
def run_benchmarks(context_len=2048):
    # Setup
    B, H = 16, 16
    MAX_BLOCKS = context_len // BLOCK_SIZE
    D = HEAD_DIM
    pool_size = max(5000, MAX_BLOCKS * B)
    scale = 1.0 / (D ** 0.5)

    print(f"Setting up benchmark: B={B}, H={H}, D={D}, Context Len={MAX_BLOCKS*BLOCK_SIZE}")

    # Data Initialization
    t_q = torch.randn(B, H, D, device='cuda', dtype=torch.float32)
    t_k = torch.randn(pool_size, H, D, BLOCK_SIZE, device='cuda', dtype=torch.float32)
    t_v = torch.randn(pool_size, H, D, BLOCK_SIZE, device='cuda', dtype=torch.float32)
    t_bt = torch.randint(0, pool_size, (B, MAX_BLOCKS), device='cuda', dtype=torch.int32)
    
    # ----------------------------------------
    # Correctness Check
    # ----------------------------------------
    print("Verifying correctness...")
    
    # 1. Run PyTorch Reference
    ref_out = pytorch_paged_attention(t_q, t_k, t_v, t_bt, scale)
    
    # 2. Run cuTile
    cp_q = cp.asarray(t_q); cp_k = cp.asarray(t_k); cp_v = cp.asarray(t_v); cp_bt = cp.asarray(t_bt)
    cp_out = cp.zeros_like(cp_q)
    cp_idx_dim = cp.arange(D, dtype=cp.int32).reshape(D, 1)
    cp_idx_blk = cp.arange(BLOCK_SIZE, dtype=cp.int32).reshape(1, BLOCK_SIZE)
    neg_inf = -float('inf')
    
    stream = cp.cuda.get_current_stream()
    ct.launch(stream, (B, H, 1), paged_attention_kernel, 
              (cp_q, cp_k, cp_v, cp_bt, cp_out, cp_idx_dim, cp_idx_blk, scale, neg_inf, BLOCK_SIZE, D, MAX_BLOCKS))
    stream.synchronize()
    
    # 3. Run Triton
    out_tri = torch.empty_like(t_q)
    grid = (B, H)
    triton_paged_attn_kernel[grid](
        t_q, t_k, t_v, t_bt, out_tri,
        t_q.stride(0), t_q.stride(1), t_q.stride(2),
        t_k.stride(0), t_k.stride(1), t_k.stride(2), t_k.stride(3),
        t_bt.stride(0), t_bt.stride(1),
        out_tri.stride(0), out_tri.stride(1), out_tri.stride(2),
        scale, MAX_BLOCKS, BLOCK_SIZE, D
    )
    torch.cuda.synchronize()

    # Compare
    cu_out_torch = torch.as_tensor(cp_out, device='cuda')
    
    diff_cu = torch.max(torch.abs(ref_out - cu_out_torch)).item()
    diff_tri = torch.max(torch.abs(ref_out - out_tri)).item()
    
    print(f"Max Diff (PyTorch vs cuTile): {diff_cu:.6f}")
    print(f"Max Diff (PyTorch vs Triton): {diff_tri:.6f}")
    assert diff_cu < 1e-2, "cuTile implementation incorrect!"
    assert diff_tri < 1e-2, "Triton implementation incorrect!"
    print(">> Correctness passed.")

    # ----------------------------------------
    # Profiling
    # ----------------------------------------
    print("\nRunning Profiler (1000 iter)...")
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    def benchmark_func(name, func, args, iters=1000):
        # Warmup
        for _ in range(10): 
            func(*args)
        torch.cuda.synchronize()
        
        start_event.record()
        for _ in range(iters):
            func(*args)
        end_event.record()
        torch.cuda.synchronize()
        
        ms = start_event.elapsed_time(end_event) / iters
        print(f"{name: <25}: {ms*1000:.2f} us")

    # 1. PyTorch
    benchmark_func("PyTorch (Reference)", pytorch_paged_attention, (t_q, t_k, t_v, t_bt, scale))

    # 2. cuTile wrapper
    def run_cutile():
        ct.launch(stream, (B, H, 1), paged_attention_kernel, 
              (cp_q, cp_k, cp_v, cp_bt, cp_out, cp_idx_dim, cp_idx_blk, scale, neg_inf, BLOCK_SIZE, D, MAX_BLOCKS))
    benchmark_func("cuTile", run_cutile, ())

    # 3. Triton wrapper
    def run_triton():
        triton_paged_attn_kernel[grid](
            t_q, t_k, t_v, t_bt, out_tri,
            t_q.stride(0), t_q.stride(1), t_q.stride(2),
            t_k.stride(0), t_k.stride(1), t_k.stride(2), t_k.stride(3),
            t_bt.stride(0), t_bt.stride(1),
            out_tri.stride(0), out_tri.stride(1), out_tri.stride(2),
            scale, MAX_BLOCKS, BLOCK_SIZE, D
        )
    benchmark_func("Triton", run_triton, ())

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Paged Attention Benchmark")
    parser.add_argument("--context-len", type=int, default=2048,
                        help="Context length in tokens (must be multiple of 16, default: 2048)")
    args = parser.parse_args()
    assert args.context_len % BLOCK_SIZE == 0, f"Context length must be a multiple of {BLOCK_SIZE}"
    run_benchmarks(context_len=args.context_len)