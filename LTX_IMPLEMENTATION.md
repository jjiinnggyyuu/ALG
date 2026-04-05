# LTX-Video ALG Implementation

> **Note**: This is an unofficial community implementation of ALG for LTX-Video.
> The official implementation is listed as "coming soon" in the original repository.

## Overview

LTX-Video uses a different I2V conditioning mechanism compared to Wan and CogVideoX.
Instead of channel-wise concatenation, LTX-Video fixes the first frame by setting
its timestep to 0 via `conditioning_mask`.

ALG is applied by replacing the first frame latent with a low-pass filtered version
during early denoising steps, then switching back to the original latent at later steps.

## How to Run

**ALG enabled (more dynamic):**
```bash
python run.py \
  --config ./configs/ltx_alg.yaml \
  --image_path ./assets/city.png \
  --prompt "A car chase through narrow city streets at night." \
  --output_path city_ltx_alg.mp4
```

**ALG disabled (baseline CFG):**
```bash
python run.py \
  --config ./configs/ltx_default.yaml \
  --image_path ./assets/city.png \
  --prompt "A car chase through narrow city streets at night." \
  --output_path city_ltx_default.mp4
```

## Implementation Details

### LTX-Video Conditioning Mechanism
```
conditioning_mask = [1, 0, 0, 0, ...]  ← 첫 프레임만 1
timestep = t * (1 - conditioning_mask) ← 첫 프레임 timestep=0
→ DiT가 첫 프레임을 "완성된 프레임"으로 인식
→ 이미지 그대로 유지
```

### ALG Application
```
Early steps (t < t_trans):
  first_frame = low_pass_filter(image_latent)
  → shortcut 방지, 동적 움직임 생성

Later steps (t >= t_trans):
  first_frame = original image_latent
  → 이미지 디테일 복원
```

## Tested Environment
- GPU: RTX 3080 Ti 12GB
- CUDA: 12.8
- PyTorch: 2.5.1
- Ubuntu: 22.04

## Limitations
- Tested on limited hardware (12GB VRAM)
- Results may improve significantly on higher-end GPUs (A100/H100)