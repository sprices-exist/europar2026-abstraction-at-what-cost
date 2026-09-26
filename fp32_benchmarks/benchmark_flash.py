import math
import torch
import torch.nn.functional as F
import cuda.tile as ct
import triton
import triton.language as tl

# ==========================================
# 1. cuTile Flash Kernel
# ==========================================
INV_LOG_2 = 1.44269504
ConstInt = ct.Constant[int]

@ct.kernel
def fmha_kernel(
    Q: ct.Array, K: ct.Array, V: ct.Array, Out: ct.Array,
    qk_scale: float, input_pos: int,
    TILE_D: ConstInt, H: ConstInt, TILE_M: ConstInt, TILE_N: ConstInt,
    QUERY_GROUP_SIZE: ConstInt,
):
    bid_x, bid_y = ct.bid(0), ct.bid(1)
    batch_idx, head_idx = bid_y // H, bid_y % H
    off_kv_h = head_idx // QUERY_GROUP_SIZE
    scale_factor = qk_scale * INV_LOG_2

    offs_m = bid_x * TILE_M + ct.arange(TILE_M, dtype=ct.int32)
    offs_m = (offs_m + input_pos)[:, None]
    
    m_i = ct.full((TILE_M, 1), -math.inf, dtype=ct.float32)
    l_i = ct.full((TILE_M, 1), 0.0, dtype=ct.float32)
    acc = ct.full((TILE_M, TILE_D), 0.0, dtype=ct.float32)

    q = ct.load(Q, index=(batch_idx, head_idx, bid_x, 0), shape=(1, 1, TILE_M, TILE_D))
    q = ct.reshape(q, (TILE_M, TILE_D))
    
    k_seqlen = K.shape[2]
    Tc = ct.cdiv(k_seqlen, TILE_N)

    for j in range(0, Tc):
        k = ct.load(K, index=(batch_idx, off_kv_h, 0, j), shape=(1, 1, TILE_D, TILE_N), order=(0, 1, 3, 2))
        k = k.reshape((TILE_D, TILE_N))
        
        qk = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)
        qk = ct.mma(q, k, qk)
        
        row_max = ct.max(qk, axis=-1, keepdims=True)
        m_ij = ct.maximum(m_i, row_max * scale_factor)
        p = ct.exp2(qk * scale_factor - m_ij)
        l_ij = ct.sum(p, axis=-1, keepdims=True)
        
        alpha = ct.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha
        
        v = ct.load(V, index=(batch_idx, off_kv_h, j, 0), shape=(1, 1, TILE_N, TILE_D))
        v = v.reshape((TILE_N, TILE_D))
        acc = ct.mma(p.astype(Q.dtype), v, acc)
        m_i = m_ij

    acc = acc / l_i
    acc = acc.reshape((1, 1, TILE_M, TILE_D)).astype(Out.dtype)
    ct.store(Out, index=(batch_idx, head_idx, bid_x, 0), tile=acc)

# ==========================================
# 2. Triton Flash Kernel
# ==========================================
@triton.jit
def _triton_fwd_kernel(
    Q, K, V, sm_scale, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_on,
    H, N_CTX, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    
    q_offset = off_z * stride_qz + off_h * stride_qh
    k_offset = off_z * stride_kz + off_h * stride_kh
    v_offset = off_z * stride_vz + off_h * stride_vh
    o_offset = off_z * stride_oz + off_h * stride_oh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)
    
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    
    offs_n = tl.arange(0, BLOCK_N)
    k_ptrs = K + k_offset + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kk
    v_ptrs = V + v_offset + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
    
    for start_n in range(0, tl.cdiv(N_CTX, BLOCK_N)):
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)
        qk = tl.dot(q, k) * sm_scale
        
        m_ij = tl.max(qk, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])
        
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_i_new
        
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn
    
    acc = acc / l_i[:, None]
    o_ptrs = Out + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < N_CTX)

# ==========================================
# 3. Benchmark Harness
# ==========================================
def run_benchmarks(S=4096):
    B, H, D = 2, 8, 64
    print(f"Flash Attention Benchmark: [B={B}, H={H}, S={S}, D={D}]")

    dtype = torch.float16
    device = "cuda"

    q = torch.randn(B, H, S, D, dtype=dtype, device=device)
    k = torch.randn(B, H, S, D, dtype=dtype, device=device)
    v = torch.randn(B, H, S, D, dtype=dtype, device=device)
    out = torch.empty_like(q)
    
    sm_scale = 1.0 / math.sqrt(D)
    
    # -------------------------------------------------
    # Helper for timing
    # -------------------------------------------------
    def benchmark_func(name, func, iters=20):
        # Warmup
        for _ in range(5):
            func()
        torch.cuda.synchronize()
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(iters):
            func()
        end.record()
        torch.cuda.synchronize()
        
        avg_time = start.elapsed_time(end) / iters
        print(f"{name:<20} : {avg_time:.4f} ms")

    # -------------------------------------------------
    # 1. PyTorch SDPA
    # -------------------------------------------------
    def run_sdpa():
        # Uses Flash Attention backend internally where possible
        F.scaled_dot_product_attention(q, k, v, scale=sm_scale)
        
    benchmark_func("PyTorch SDPA", run_sdpa)

    # -------------------------------------------------
    # 2. cuTile Flash
    # -------------------------------------------------
    TILE_M, TILE_N = 64, 32
    grid_ct = (math.ceil(S / TILE_M), B * H, 1)
    
    def run_cutile():
        ct.launch(torch.cuda.current_stream(), grid_ct, fmha_kernel,
                  (q, k, v, out, sm_scale, 0, D, H, TILE_M, TILE_N, 1))
                  
    benchmark_func("cuTile Flash", run_cutile)

    # -------------------------------------------------
    # 3. Triton Flash
    # -------------------------------------------------
    grid_tri = (triton.cdiv(S, 128), B * H)
    
    def run_triton():
        _triton_fwd_kernel[grid_tri](
            q, k, v, sm_scale, out,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            H, S, BLOCK_M=128, BLOCK_N=64, BLOCK_DMODEL=D
        )
        
    benchmark_func("Triton Flash", run_triton)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Flash Attention Benchmark")
    parser.add_argument("--seq-len", type=int, default=4096,
                        help="Sequence length (default: 4096)")
    args = parser.parse_args()
    run_benchmarks(S=args.seq_len)