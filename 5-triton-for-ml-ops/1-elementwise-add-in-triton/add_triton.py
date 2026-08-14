import torch, triton
import triton.language as tl

@triton.jit
def add_kernel(a_ptr, b_ptr, c_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(a_ptr + offs, mask=mask)
    b = tl.load(b_ptr + offs, mask=mask)
    tl.store(c_ptr + offs, a + b, mask=mask)

def triton_add(a, b):
    c = torch.empty_like(a)
    n = a.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
    add_kernel[grid](a, b, c, n, BLOCK=1024)
    return c

if __name__ == "__main__":
    n = 1 << 24
    a, b = (torch.randn(n, device="cuda") for _ in range(2))
    assert torch.allclose(triton_add(a, b), a + b); print("PASS")
    for name, fn in [("triton", lambda: triton_add(a, b)), ("torch", lambda: a + b)]:
        ms = triton.testing.do_bench(fn)
        print(f"{name}: {ms:.3f} ms, {3 * n * 4 / ms / 1e6:.0f} GB/s")