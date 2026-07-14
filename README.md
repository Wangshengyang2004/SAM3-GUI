# SAM3.1 GUI/API

**English** | [简体中文](docs/README_CN.md)

Native GUI and HTTP API for **SAM 3.1 Object Multiplex** video and image segmentation. The app supports text, point, and box prompts through the latest SAM 3.1 request API from `facebookresearch/sam3`.

## Features

- **Native SAM 3.1 support**: Uses the current `facebookresearch/sam3` Object Multiplex API.
- **Text prompting**: Segment objects using natural language such as `person`, `car`, or `red shoe`.
- **Point prompts**: Interactive positive/negative point refinement.
- **Box prompts**: Draw bounding boxes as geometric prompt guidance.
- **Video tracking**: Multi-object tracking across frame sequences with forward, backward, or bidirectional propagation.
- **HTTP API**: FastAPI routes for sessions, prompts, propagation, object removal, and image segmentation.

## Installation

### Prerequisites

- Python 3.12 or higher
- PyTorch 2.7 or higher with CUDA 12.6 or higher recommended
- CUDA-compatible GPU for practical inference
- **FFmpeg** for video processing: `sudo apt-get install ffmpeg` on Ubuntu/Debian or `brew install ffmpeg` on macOS

### Install SAM3

Install [SAM3](https://github.com/facebookresearch/sam3) first:

```bash
conda create -n sam3 python=3.12
conda activate sam3
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .

# Optional: Install additional dependencies for example notebooks or development
# For running example notebooks
pip install -e ".[notebooks]"

# For development
pip install -e ".[train,dev]"
```

**Note for Blackwell RTX 50X0 GPUs:** these GPUs may require torchvision to be compiled from source if compatible prebuilt wheels are unavailable. See [docs/blackwell_support.md](docs/blackwell_support.md).

Install GUI/API dependencies:

```bash
cd SAM3-GUI
pip install -e .
```

Install SAM3 first so the environment uses the correct CUDA-enabled PyTorch build. The package metadata also declares the direct Python dependencies needed by the GUI and API.

## Checkpoint

SAM3-GUI uses `sam3.1_multiplex.pt` from the SAM 3.1 checkpoint repo. Hugging Face is the default source:

```bash
python -m tools.download_model --source huggingface
```

The Hugging Face checkpoint repo is gated, so request access first and authenticate locally:

```bash
hf auth login
```

If Hugging Face is unavailable, use the ModelScope mirror:

```bash
python -m tools.download_model --source modelscope
```

Both routes store `sam3.1_multiplex.pt` under `~/sam3/model` by default. If your ModelScope mirror uses a different repo id, pass it explicitly:

```bash
python -m tools.download_model --source modelscope --modelscope_model_id <repo-id>
```

The default ModelScope repo id is `facebook/sam3.1`, matching the ModelScope page for SAM 3.1. If no local checkpoint is provided, the backend lets SAM3 download `facebook/sam3.1` through its native loader on first model load.

## Start

```bash
sam3-gui data.root_dir=data_root server.port=8890
```

Common Hydra overrides:

- `sam.checkpoint_path=/path/to/sam3.1_multiplex.pt`
- `sam.gpus=[0]`
- `sam.use_fa3=true`
- `data.vid_name=videos data.img_name=images data.mask_name=masks`

`sam.use_fa3=true` runs a small FlashAttention 3 CUDA preflight before model loading. On RTX 5090/Blackwell, keep the default `false` unless FA3 was built with compatible sm_120 kernels. See [docs/blackwell_support.md](docs/blackwell_support.md).

Defaults are composed from [sam3_gui/conf/config.yaml](sam3_gui/conf/config.yaml) and the `server`, `data`, and `sam` groups. Legacy flags such as `--root_dir`, `--port`, and `--use_fa3` still work, but new changes should prefer Hydra overrides.

The server binds to `127.0.0.1` by default. For a trusted LAN or Tailscale deployment, explicitly set `server.name=0.0.0.0` and apply network-level access controls.

The Gradio UI is mounted at `/`. API docs are available at `/docs`.

## Data Layout

```text
data_root/
├── videos/     # source videos
├── images/     # extracted frame folders
└── masks/      # saved mask outputs
```

## API

HTTP routes are documented in [docs/API.md](docs/API.md). OpenAPI is available at `/docs` when the server is running.

Quick start:

```bash
curl http://127.0.0.1:8890/api/health

curl -X POST http://127.0.0.1:8890/api/images/segment \
  -F file=@/path/to/image.jpg \
  -F text=person
```

For session-based video tracking, prompt/propagate examples, and the full `/api/images/segment` parameter reference, see [docs/API.md](docs/API.md).

## Tests

Run the offline suite first. It covers the API surface, launcher/configuration, checkpoint guardrails, mask serialization, and UI handler request shapes without loading the checkpoint:

```bash
pip install -e ".[test]"
python -m pytest -q tests
```

Tests that need real SAM 3.1 inference are skipped unless `~/sam3/model/sam3.1_multiplex.pt` exists. To run the full integration path, download or place the checkpoint, then run the same command. To let SAM3 download during tests instead of using a local file:

```bash
SAM3_ALLOW_HF_DOWNLOAD=1 python -m pytest -q tests
```

Use `SAM3_CHECKPOINT_PATH=/path/to/sam3.1_multiplex.pt` for a custom checkpoint location.

Focused real-inference checks used during development:

```bash
SAM3_CHECKPOINT_PATH=~/sam3/model/sam3.1_multiplex.pt \
python -m pytest -q \
tests/test_integration_sam31_gui_api.py::test_native_sam31_real_mp4_text_box_prompt_smoke -s
```

This test decodes a real `.mp4`, adds a `text + box` prompt, and runs propagation through the native SAM 3.1 checkpoint.

## UI Workflow

Video mode supports text, point, and box prompts on frame folders or extracted video frames, then tracks with SAM 3.1 Object Multiplex through forward, backward, or both directions. Image mode uses the same SAM 3.1 request API by starting a single-frame session.

Saved masks are written to `{root_dir}/masks/{sequence_name}/` as color PNGs and index-mask `.npy` files.

## Docs

- [HTTP API](docs/API.md)
- [中文说明](docs/README_CN.md)
- [Changelog](docs/CHANGELOG.md)
- [Blackwell GPU notes](docs/blackwell_support.md)

## Verified Behavior

![SAM3 GUI Video Mode](docs/assets/sam3_1.png)

![SAM3 GUI Image Mode](docs/assets/sam3_2.png)

- Real image sessions detect ordinary online images such as `car` and `dog` with high confidence.
- Real `.mp4` sessions load through OpenCV frame decoding and propagate tracked objects after a successful prompt.
- If a prompt finds no objects, propagation returns no frames instead of calling the native propagation path with an empty object set.
- Hugging Face and ModelScope checkpoint download routes both target `sam3.1_multiplex.pt`.

## Acknowledgments

This project builds on Meta's [SAM3](https://github.com/facebookresearch/sam3) implementation and checkpoint releases.
