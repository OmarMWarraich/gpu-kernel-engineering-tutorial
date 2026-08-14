import numpy as np
N = 1024
i = np.arange(N * N, dtype=np.float32)
A = np.sin(i * 0.001).reshape(N, N).astype(np.float32)
B = np.cos(i * 0.001).reshape(N, N).astype(np.float32)
C_ref = A @ B
C_gpu = np.fromfile("C_gpu.bin", dtype=np.float32).reshape(N, N)
rel = np.abs(C_gpu - C_ref).max() / np.abs(C_ref).max()
print("PASS" if rel < 1e-3 else f"FAIL rel={rel}")