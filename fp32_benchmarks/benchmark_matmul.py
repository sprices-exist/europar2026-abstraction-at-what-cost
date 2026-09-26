import math
import torch
import triton
import triton.language as tl
import cuda.tile as ct
import warp as wp

# Constants
ConstInt = ct.Constant[int]
zero_pad = ct.PaddingMode.ZERO
GROUP_SIZE_M = 8

# ==========================================
# 1. cuTile Matmul
# ==========================================
def swizzle_2d(M, N, tm, tn, group_size_m):
    bid = ct.bid(0)
    num_bid_m = ct.cdiv(M, tm)
    num_bid_n = ct.cdiv(N, tn)
    num_bid_in_group = group_size_m * num_bid_n
    group_id = bid // num_bid_in_group
    first_bid_m = group_id * group_size_m
    group_size_m_actual = min(num_bid_m - first_bid_m, group_size_m)
    bid_m = first_bid_m + (bid % group_size_m_actual)
    bid_n = (bid % num_bid_in_group) // group_size_m_actual
    return bid_m * tm, bid_n * tn

@ct.kernel
def matmul_kernel(A, B, C, tm: ConstInt, tn: ConstInt, tk: ConstInt):
    M, N = A.shape[0], B.shape[1]
    
    # Re-calc swizzle logic for tile index
    bid = ct.bid(0)
    num_bid_m = ct.cdiv(M, tm)
    num_bid_n = ct.cdiv(N, tn)
    num_bid_in_group = GROUP_SIZE_M * num_bid_n
    group_id = bid // num_bid_in_group
    first_bid_m = group_id * GROUP_SIZE_M
    group_size_m_actual = min(num_bid_m - first_bid_m, GROUP_SIZE_M)
    
    tile_idx_m = first_bid_m + (bid % group_size_m_actual)
    tile_idx_n = (bid % num_bid_in_group) // group_size_m_actual

    num_tiles_k = ct.num_tiles(A, axis=1, shape=(tm, tk))
    accumulator = ct.full((tm, tn), 0, dtype=ct.float32)

    for k in range(num_tiles_k):
        a = ct.load(A, index=(tile_idx_m, k), shape=(tm, tk), padding_mode=zero_pad)
        b = ct.load(B, index=(k, tile_idx_n), shape=(tk, tn), padding_mode=zero_pad)
        accumulator = ct.mma(a, b, accumulator)

    accumulator = ct.astype(accumulator, C.dtype)
    ct.store(C, index=(tile_idx_m, tile_idx_n), tile=accumulator)

# ==========================================
# 2. Triton Matmul
# ==========================================
@triton.jit
def triton_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        
    c = accumulator.to(tl.float16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, c, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))

# ==========================================
# 3. Warp Matmul
# ==========================================
def create_warp_gemm(m, n, k):
    TILE_M = m
    TILE_N = n
    TILE_K = k

    # Inputs are float16 (to use Tensor Cores/mixed precision), 
    # Output must be float32 to match the accumulator type in tile_store.
    @wp.kernel
    def gemm(A: wp.array2d(dtype=wp.float16), 
             B: wp.array2d(dtype=wp.float16), 
             output: wp.array2d(dtype=wp.float32)): 
        i, j = wp.tid()
        
        # Accumulate in float32 for precision
        sum = wp.tile_zeros(shape=(TILE_M, TILE_N), dtype=wp.float32)

        count = A.shape[1] // TILE_K

        for k in range(count):
            a = wp.tile_load(A, shape=(TILE_M, TILE_K), offset=(i * TILE_M, k * TILE_K))
            b = wp.tile_load(B, shape=(TILE_K, TILE_N), offset=(k * TILE_K, j * TILE_N))
            
            # tile_matmul supports f16 inputs + f32 accum
            wp.tile_matmul(a, b, sum)

        # Store result (requires output to be float32 to match sum)
        wp.tile_store(output, sum, offset=(i * TILE_M, j * TILE_N))

    return gemm

def run_benchmarks(SIZE=2048):
    wp.init()
    
    print(f"GEMM Benchmark: {SIZE}x{SIZE}, FP16")
    A = torch.randn((SIZE, SIZE), device='cuda', dtype=torch.float16)
    B = torch.randn((SIZE, SIZE), device='cuda', dtype=torch.float16)
    C = torch.empty((SIZE, SIZE), device='cuda', dtype=torch.float16)
    
    # cuTile
    tm, tn, tk = 64, 64, 32
    grid_size = math.ceil(SIZE / tm) * math.ceil(SIZE / tn)
    print("Running cuTile Matmul...")
    for _ in range(5):
        ct.launch(torch.cuda.current_stream(), (grid_size, 1, 1), matmul_kernel, (A, B, C, tm, tn, tk))
    
    # Triton
    grid = lambda META: (triton.cdiv(SIZE, META['BLOCK_SIZE_M']) * triton.cdiv(SIZE, META['BLOCK_SIZE_N']), )
    print("Running Triton Matmul...")
    for _ in range(5):
        triton_matmul_kernel[grid](
            A, B, C, SIZE, SIZE, SIZE,
            A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1),
            BLOCK_SIZE_M=128, BLOCK_SIZE_N=256, BLOCK_SIZE_K=64, GROUP_SIZE_M=8
        )

    # Warp
    tile_m, tile_n, tile_k = 64, 64, 32
    block_dim = 128
    warp_kernel = create_warp_gemm(tile_m, tile_n, tile_k)
    
    # Convert Torch tensors to Warp arrays
    # A and B are float16 (zero-copy from torch)
    A_wp = wp.from_torch(A)
    B_wp = wp.from_torch(B)
    
    # Create a float32 output buffer for Warp to avoid TypeError in tile_store
    C_wp = wp.zeros((SIZE, SIZE), dtype=wp.float32, device='cuda')
    
    print("Running Warp Matmul...")
    for _ in range(5):
        wp.launch_tiled(
            kernel=warp_kernel,
            dim=[SIZE // tile_m, SIZE // tile_n],
            inputs=[A_wp, B_wp, C_wp],
            block_dim=block_dim
        )
    
    # PyTorch
    print("Running PyTorch Matmul...")
    for _ in range(5):
        torch.matmul(A, B)

    torch.cuda.synchronize()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GEMM Benchmark")
    parser.add_argument("--size", type=int, default=2048,
                        help="Square matrix dimension (default: 2048)")
    args = parser.parse_args()
    run_benchmarks(SIZE=args.size)