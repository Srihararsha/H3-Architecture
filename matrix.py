import numpy as np
import re

# =========================================================
# READ FILES
# =========================================================

with open("u.txt", "r") as f:
    u_text = f.read()

with open("wq.txt", "r") as f:
    wq_text = f.read()

with open("bq.txt", "r") as f:
    bq_text = f.read()

print("FILES LOADED")

# =========================================================
# NUMBER PARSER
# Handles:
# - floats
# - negatives
# - scientific notation
# =========================================================

def parse_numbers(text, dtype=np.float16):
    pattern = r'[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?'
    numbers = re.findall(pattern, text)
    return np.array([float(x) for x in numbers], dtype=dtype)

# =========================================================
# PARSE u
# =========================================================

u_values = parse_numbers(u_text, dtype=np.float16)

print("\nParsed u values:", len(u_values))

expected_u = 1 * 3 * 768

if len(u_values) != expected_u:
    print("ERROR: u tensor size mismatch")
    print("Expected:", expected_u)
    print("Got     :", len(u_values))
    exit()

u = u_values.reshape(1, 3, 768)

print("u shape:", u.shape)

# =========================================================
# PARSE Wq
# =========================================================

wq_values = parse_numbers(wq_text, dtype=np.float16)

print("\nParsed Wq values:", len(wq_values))

expected_wq = 768 * 768

if len(wq_values) != expected_wq:
    print("ERROR: Wq tensor size mismatch")
    print("Expected:", expected_wq)
    print("Got     :", len(wq_values))
    exit()

Wq = wq_values.reshape(768, 768)

print("Wq shape:", Wq.shape)

# =========================================================
# PARSE bq
# =========================================================

bq_values = parse_numbers(bq_text, dtype=np.float16)

print("\nParsed bq values:", len(bq_values))

expected_bq = 768

if len(bq_values) != expected_bq:
    print("ERROR: bq tensor size mismatch")
    print("Expected:", expected_bq)
    print("Got     :", len(bq_values))
    exit()

bq = bq_values.reshape(768)

print("bq shape:", bq.shape)

# =========================================================
# MATRIX MULTIPLICATION
# Q = u @ Wq^T + bq
# =========================================================

# FP32 accumulation improves numerical stability
Q = np.matmul(
    u.astype(np.float16),
    Wq.T.astype(np.float16)
)
print(Q.shape)
print("\nFirst token first 10 FP32 values after multiplication:")
print(Q[0, 0, :10])

Q = Q + bq.astype(np.float16)

# Optional:
# convert back to fp16 to mimic transformer output
Q_fp16 = Q.astype(np.float16)

print("\nMATMUL DONE")

print("Q shape:", Q.shape)

# =========================================================
# OUTPUT
# =========================================================

np.set_printoptions(
    precision=10,
    suppress=False,
    linewidth=200
)

print("\nFirst token first 10 FP32 values:")
print(Q[0, 0, :10])

print("\nFirst token first 10 FP16 values:")
print(Q_fp16[0, 0, :10])

print("\nFirst scalar:")
print("FP32 :", Q[0, 0, 0])
print(repr(Q_fp16[0, 0, 0]))
print("FP16 :")
print(float(Q_fp16[0, 0, 0]))
# print("FP16 :", Q_fp16[0, 0, 0])