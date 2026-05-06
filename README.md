# SAM3.1 GUI/API

Native GUI and HTTP API for **SAM 3.1 Object Multiplex** video and image segmentation. The app intentionally supports only the latest SAM 3.1 path from `facebookresearch/sam3`; older SAM3 entry points and compatibility shims are not used.

## Requirements

- Python 3.12+
- PyTorch 2.7+ with CUDA 12.6+ recommended
- Latest `/home/wsy/sam3` installed editable:

```bash
cd /home/wsy/sam3
pip install -e .
```

Install GUI/API dependencies:

```bash
cd /home/wsy/SAM3-GUI
pip install -r requirements.txt
```

## Checkpoint

SAM3-GUI uses `sam3.1_multiplex.pt` from the SAM 3.1 checkpoint repo. Hugging Face is the default source:

```bash
python download_model.py --source huggingface
```

The Hugging Face checkpoint repo is gated, so request access first and authenticate locally:

```bash
hf auth login
```

If Hugging Face is unavailable, use the ModelScope mirror:

```bash
python download_model.py --source modelscope
```

Both routes store `sam3.1_multiplex.pt` under `~/sam3/model` by default. If your ModelScope mirror uses a different repo id, pass it explicitly:

```bash
python download_model.py --source modelscope --modelscope_model_id <repo-id>
```

The default ModelScope repo id is `facebook/sam3.1`, matching the ModelScope page for SAM 3.1. If no local checkpoint is provided, the backend lets SAM3 download `facebook/sam3.1` through its native loader on first model load.

## Start

```bash
python cli.py --root_dir data_root --server_name 0.0.0.0 --port 8890
```

Options:

- `--checkpoint_path`: local `sam3.1_multiplex.pt` path
- `--gpus`: comma-separated CUDA IDs; the SAM3.1 backend uses the first ID
- `--vid_name`, `--img_name`, `--mask_name`: data subdirectory names

The Gradio UI is mounted at `/`. API docs are available at `/docs`.

## Data Layout

```text
data_root/
├── videos/     # source videos
├── images/     # extracted frame folders
└── masks/      # saved mask outputs
```

## API

Health:

```bash
curl http://127.0.0.1:8890/api/health
```

Start a video/frame-folder session:

```bash
curl -X POST http://127.0.0.1:8890/api/sessions \
  -H 'Content-Type: application/json' \
  -d '{"resource_path":"/home/wsy/SAM3-GUI/data_root/images/Cam1_color"}'
```

Add a text prompt:

```bash
curl -X POST http://127.0.0.1:8890/api/prompts \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"SESSION","frame_index":0,"text":"person","include_masks":false}'
```

Add point prompts using normalized coordinates:

```bash
curl -X POST http://127.0.0.1:8890/api/prompts \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"SESSION","frame_index":0,"obj_id":0,"points":[[0.5,0.5]],"point_labels":[1]}'
```

Add a box prompt using normalized `[x, y, width, height]`:

```bash
curl -X POST http://127.0.0.1:8890/api/prompts \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"SESSION","frame_index":0,"bounding_boxes":[[0.2,0.2,0.4,0.5]],"bounding_box_labels":[1]}'
```

For SAM 3.1 Object Multiplex, text grounding is the reliable way to create new objects. Box prompts are accepted as geometric guidance, but pure box-only prompts may return no objects on some videos. Use `text` plus `bounding_boxes` when starting an object from a video frame:

```bash
curl -X POST http://127.0.0.1:8890/api/prompts \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"SESSION","frame_index":0,"text":"vehicle","bounding_boxes":[[0.4094,0.8184,0.3274,0.1816]],"bounding_box_labels":[1],"include_masks":false}'
```

Propagate:

```bash
curl -X POST http://127.0.0.1:8890/api/propagate \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"SESSION","propagation_direction":"both","include_masks":false}'
```

Remove an object:

```bash
curl -X POST http://127.0.0.1:8890/api/objects/remove \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"SESSION","obj_id":0}'
```

Close a session:

```bash
curl -X DELETE http://127.0.0.1:8890/api/sessions/SESSION
```

Segment one uploaded image:

```bash
curl -X POST http://127.0.0.1:8890/api/images/segment \
  -F file=@/path/to/image.jpg \
  -F text=person
```

Mask outputs are COCO RLE when `include_masks` is true.

## Tests

Run the offline suite first. It covers the API surface, launcher/configuration, checkpoint guardrails, mask serialization, and UI handler request shapes without loading the checkpoint:

```bash
cd /home/wsy/SAM3-GUI
/home/wsy/miniconda3/envs/sam3/bin/python -m pytest -q tests
```

Tests that need real SAM 3.1 inference are skipped unless `~/sam3/model/sam3.1_multiplex.pt` exists. To run the full integration path, download or place the checkpoint, then run the same command. To let SAM3 download during tests instead of using a local file:

```bash
SAM3_ALLOW_HF_DOWNLOAD=1 /home/wsy/miniconda3/envs/sam3/bin/python -m pytest -q tests
```

Use `SAM3_CHECKPOINT_PATH=/path/to/sam3.1_multiplex.pt` for a custom checkpoint location.

Focused real-inference checks used during development:

```bash
SAM3_CHECKPOINT_PATH=/home/wsy/sam3/model/sam3.1_multiplex.pt \
/home/wsy/miniconda3/envs/sam3/bin/python -m pytest -q \
tests/test_integration_sam31_gui_api.py::test_native_sam31_real_mp4_text_box_prompt_smoke -s
```

This test decodes a real `.mp4`, adds a `text + box` prompt, and runs propagation through the native SAM 3.1 checkpoint.

## UI Workflow

Video mode supports text, point, and box prompts on frame folders or extracted video frames, then tracks with SAM 3.1 Object Multiplex through forward, backward, or both directions. Image mode uses the same SAM 3.1 request API by starting a single-frame session.

Saved masks are written to `{root_dir}/masks/{sequence_name}/` as color PNGs and index-mask `.npy` files.

## Verified Behavior

- Real image sessions detect ordinary online images such as `car` and `dog` with high confidence.
- Real `.mp4` sessions load through OpenCV frame decoding and propagate tracked objects after a successful prompt.
- If a prompt finds no objects, propagation returns no frames instead of calling the native propagation path with an empty object set.
- Hugging Face and ModelScope checkpoint download routes both target `sam3.1_multiplex.pt`.
