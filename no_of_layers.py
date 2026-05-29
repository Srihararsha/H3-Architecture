import torch
import re

ckpt_path = "H3-125M/model.pt"

# IMPORTANT FIX
ckpt = torch.load(
    ckpt_path,
    map_location="cpu",
    weights_only=False
)

# Handle state_dict wrapping
if isinstance(ckpt, dict) and "state_dict" in ckpt:
    state_dict = ckpt["state_dict"]
else:
    state_dict = ckpt

print("=" * 80)
print("CHECKPOINT ANALYSIS")
print("=" * 80)

print(f"\nCheckpoint Type: {type(state_dict)}")

if not isinstance(state_dict, dict):
    print("Checkpoint is not a standard state_dict.")
    exit()

print(f"Number of tensors: {len(state_dict)}")

# ---------------------------------------------------------
# Print first few parameter names
# ---------------------------------------------------------

print("\nFIRST 50 PARAMETER NAMES:\n")

for i, key in enumerate(state_dict.keys()):
    print(f"{i+1:02d}. {key}")

    if i >= 49:
        break

# ---------------------------------------------------------
# Detect layers
# ---------------------------------------------------------

layer_patterns = [
    r"layers\.(\d+)",
    r"h\.(\d+)",
    r"block[s]?\.(\d+)",
    r"encoder\.layer\.(\d+)",
    r"decoder\.layer\.(\d+)"
]

layer_ids = set()

for key in state_dict.keys():

    for pattern in layer_patterns:

        match = re.search(pattern, key)

        if match:
            layer_ids.add(int(match.group(1)))

print("\n" + "=" * 80)

if layer_ids:

    sorted_layers = sorted(layer_ids)

    print("Detected Layer IDs:")
    print(sorted_layers)

    print(f"\nEstimated Number of Layers: {len(sorted_layers)}")
    print(f"Highest Layer Index: {max(sorted_layers)}")

else:
    print("Could not automatically detect layers.")

print("=" * 80)

# ---------------------------------------------------------
# Total parameter count
# ---------------------------------------------------------

total_params = 0

for value in state_dict.values():

    if torch.is_tensor(value):
        total_params += value.numel()

print(f"\nTotal Parameters: {total_params:,}")

# ---------------------------------------------------------
# Print tensor shapes
# ---------------------------------------------------------

print("\nIMPORTANT TENSOR SHAPES:\n")

keywords = [
    "embedding",
    "embed",
    "q_proj",
    "k_proj",
    "v_proj",
    "attn",
    "mixer",
    "mlp",
    "fc",
    "norm"
]

count = 0

for key, value in state_dict.items():

    if any(k in key.lower() for k in keywords):

        if torch.is_tensor(value):

            print(f"{key:<80} {list(value.shape)}")

            count += 1

        if count >= 100:
            break

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)