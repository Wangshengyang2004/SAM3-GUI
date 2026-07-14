# RTX 5090 (Blackwell/sm_120) Support

## Problem

Pre-built `torchvision` wheels don't include CUDA kernels for sm_120 (RTX 5090).

```bash
# Error: "no kernel image is available for execution on the device"
# Cause: torchvision only compiled up to sm_90
```

## Solution

Compile `torchvision` from source with CUDA 12.0 support:

```bash
# 1. Clone torchvision
proxychains4 git clone https://github.com/pytorch/vision.git /tmp/vision

# 2. Install build dependencies
pip install ninja pytest wheel

# 3. Build with sm_120 support
cd /tmp/vision
export TORCH_CUDA_ARCH_LIST="12.0"  # Note: use "12.0", NOT "sm_120"
pip install -v --no-build-isolation .
```

## Verification

```bash
python -c "
from torchvision.ops import roi_align
import torch
x = torch.randn(1, 256, 64, 64).cuda()
r = torch.tensor([[0, 0, 0, 10, 10]], dtype=torch.float32).cuda()
print(roi_align(x, r, (7, 7), 1.0, 2, aligned=True).shape)
"
# Output: torch.Size([1, 256, 7, 7]) ✓
```

## Result

- `torchvision 0.26.0a0+4b0a90c` with sm_120 kernels
- Box segmentation works on RTX 5090

## FlashAttention 3

`sam.use_fa3=true` is separate from `torchvision`. On RTX 5090, the installed FA3 binary must also include sm_120 kernels. If it does not, SAM3-GUI fails early with:

```text
sam.use_fa3=true was requested, but the installed FlashAttention 3 CUDA kernel does not run on ...
```

Keep the default `sam.use_fa3=false` until this probe passes, or rebuild/replace FA3 for the local CUDA and GPU architecture.

---

**Note**: Format must be `12.0` (major.minor), not `sm_120`.
