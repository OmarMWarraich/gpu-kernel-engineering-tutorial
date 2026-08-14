import torch, triton
import triton.language as tl

@triton.jit
def fused_add_relu(x_ptr, b_ptr, y_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    v = tl.load(x_ptr + offs, mask=mask) + tl.load(b_ptr + offs, mask=mask)
    tl.store(y_ptr + offs, tl.maximum(v, 0.0), mask=mask)

n = 1 << 25
x, b = (torch.randn(n, device="cuda") for _ in range(2))
y = torch.empty_like(x)
grid = (triton.cdiv(n, 1024),)
fused_add_relu[grid](x, b, y, n, BLOCK=1024)
assert torch.allclose(y, torch.relu(x + b)); print("PASS")
t_f = triton.testing.do_bench(lambda: fused_add_relu[grid](x, b, y, n, BLOCK=1024))
t_u = triton.testing.do_bench(lambda: torch.relu(x + b))
print(f"fused {t_f:.3f} ms vs unfused {t_u:.3f} ms")