p = 0.9
for n in [2, 8, 32, 1024, float("inf")]:
    amdahl = 1 / ((1 - p) + (p / n if n != float("inf") else 0))
    gustafson = (1 - p) + p * n if n != float("inf") else float("inf")
    print(f"N={n}: Amdahl S={amdahl:.2f}, Gustafson S={gustafson}")