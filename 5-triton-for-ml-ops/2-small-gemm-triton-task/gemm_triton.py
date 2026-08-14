import torch, triton
import triton.language as tl

@triton.jit
def gemm_kernel(A, B, C, M, N, K,
                sam, sak, sbk, sbn, scm, scn,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    grid_n = tl.cdiv(N, BN)
    pid_m, pid_n = pid // grid_n, pid % grid_n
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a_ptrs = A + offs_m[:, None] * sam + offs_k[None, :] * sak
    b_ptrs = B + offs_k[:, None] * sbk + offs_n[None, :] * sbn
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k, other=0.0)
        acc = tl.dot(a, b, acc)          # tensor cores for fp16 inputs
        a_ptrs += BK * sak
        b_ptrs += BK * sbk
    c_ptrs = C + offs_m[:, None] * scm + offs_n[None, :] * scn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=mask)

def triton_matmul(a, b):
    M, K = a.shape; _, N = b.shape
    c = torch.empty((M, N), device="cuda", dtype=torch.float16)
    BM, BN, BK = 64, 64, 32
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    gemm_kernel[grid](a, b, c, M, N, K,
                      a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                      c.stride(0), c.stride(1), BM=BM, BN=BN, BK=BK)
    return c

if __name__ == "__main__":
    for n in (256, 512, 1024):
        a = torch.randn(n, n, device="cuda", dtype=torch.float16)
        b = torch.randn(n, n, device="cuda", dtype=torch.float16)
        ok = torch.allclose(triton_matmul(a, b), a @ b, rtol=1e-2, atol=1e-2)
        t_tr = triton.testing.do_bench(lambda: triton_matmul(a, b))
        t_pt = triton.testing.do_bench(lambda: a @ b)
        tf = lambda ms: 2 * n**3 / ms / 1e9
        print(f"N={n}: {'PASS' if ok else 'FAIL'} "
              f"triton {tf(t_tr):.1f} TFLOPS vs torch {tf(t_pt):.1f} TFLOPS")