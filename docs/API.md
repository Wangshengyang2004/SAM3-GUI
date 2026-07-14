# HTTP API

FastAPI serves native SAM 3.1 Object Multiplex routes under `/api/*`. Interactive docs are available at `/docs` and `/openapi.json` when the app is running.

Default local URL: `http://127.0.0.1:8890`

The default bind address is loopback-only. For an intentional remote deployment, explicitly override it and place authentication/TLS in front of the service, for example:

```bash
python cli.py server.name=0.0.0.0 server.port=8890
```

The API does not enable cross-origin access by default. Configure a trusted reverse proxy if browser clients on another origin need access; do not use wildcard CORS for a model/file-processing service.

## Health

```bash
curl http://127.0.0.1:8890/api/health
```

## Session-based video and image workflows

Use sessions when the resource already exists on the server filesystem (frame folder, video file, or single image path).

`resource_path` is resolved under the configured `data.root_dir`. Relative paths are recommended. Absolute paths are accepted only when their resolved target remains inside that root. Parent traversal and symbolic-link escapes are rejected.

### Start a session

```bash
curl -X POST http://127.0.0.1:8890/api/sessions \
  -H 'Content-Type: application/json' \
  -d '{"resource_path":"videos/example.mp4"}'
```

### Add prompts

Text:

```bash
curl -X POST http://127.0.0.1:8890/api/prompts \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"SESSION","frame_index":0,"text":"person","include_masks":false}'
```

Points (normalized `[x, y]` by default):

```bash
curl -X POST http://127.0.0.1:8890/api/prompts \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"SESSION","frame_index":0,"obj_id":0,"points":[[0.5,0.5]],"point_labels":[1]}'
```

Box (normalized `[x, y, width, height]`):

```bash
curl -X POST http://127.0.0.1:8890/api/prompts \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"SESSION","frame_index":0,"bounding_boxes":[[0.2,0.2,0.4,0.5]],"bounding_box_labels":[1]}'
```

Text plus box guidance (recommended for video frames when box-only prompts are weak):

```bash
curl -X POST http://127.0.0.1:8890/api/prompts \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"SESSION","frame_index":0,"text":"vehicle","bounding_boxes":[[0.4094,0.8184,0.3274,0.1816]],"bounding_box_labels":[1],"include_masks":false}'
```

### Propagate, remove, close

```bash
curl -X POST http://127.0.0.1:8890/api/propagate \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"SESSION","propagation_direction":"both","max_frame_num_to_track":100,"include_masks":false}'

curl -X POST http://127.0.0.1:8890/api/objects/remove \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"SESSION","obj_id":0}'

curl -X DELETE http://127.0.0.1:8890/api/sessions/SESSION
```

Session prompt responses use parallel arrays under `outputs`: `object_ids`, `probabilities`, `boxes_xywh`, and optional `masks_rle` (COCO RLE).

Frame indices and object IDs must be non-negative. Probability thresholds must be between `0` and `1`. Points use exactly two coordinates, boxes use exactly four coordinates, and label arrays must match their prompt arrays. Prompt requests require at least one non-empty text, point, or box prompt. Propagation always uses a positive frame count and is capped server-side at `1000` frames per request.

---

## Stateless single-image segmentation

`POST /api/images/segment` uploads one image and runs a single-frame prompt without exposing session management to the client. Use this for camera-frame uploads and other stateless perception loops.

### Request (multipart form)

| Field | Required | Description |
|-------|----------|-------------|
| `file` | yes | Image upload |
| `text` | one of `text` / `points` / `bounding_boxes` | Text concept prompt |
| `points` | | JSON array, e.g. `[[320,240]]` |
| `point_labels` | | JSON array, e.g. `[1]`; defaults to all `1` |
| `bounding_boxes` | | JSON array of normalized `[x,y,w,h]` |
| `bounding_box_labels` | | JSON array; defaults to all `1` |
| `rel_coordinates` | | Default `true`. Set `false` for pixel `[x,y]` points |
| `output_prob_thresh` | | Default `0.5` |
| `response_format` | | `legacy` (default) or `aspire` |
| `box_format` | | `xywh_normalized` (default) or `xywh_pixel` |
| `include_overlay` | | Default `false` |

Uploads are streamed to disk and limited to 20 MiB. The endpoint accepts single-frame BMP, JPEG, PNG, TIFF, and WebP images up to 40,000,000 pixels. Oversized uploads/images return `413`; unsupported, invalid, or animated image content returns `415`.

### Examples

Text only:

```bash
curl -X POST http://127.0.0.1:8890/api/images/segment \
  -F file=@/path/to/image.jpg \
  -F text=person
```

Text with per-instance list response:

```bash
curl -X POST http://127.0.0.1:8890/api/images/segment \
  -F file=@/path/to/image.jpg \
  -F text=bowl \
  -F response_format=aspire
```

Point prompt with pixel coordinates:

```bash
curl -X POST http://127.0.0.1:8890/api/images/segment \
  -F file=@/path/to/image.jpg \
  -F 'points=[[320,240]]' \
  -F 'point_labels=[1]' \
  -F rel_coordinates=false \
  -F response_format=aspire
```

Text plus box:

```bash
curl -X POST http://127.0.0.1:8890/api/images/segment \
  -F file=@/path/to/image.jpg \
  -F text=vehicle \
  -F 'bounding_boxes=[[0.2,0.2,0.4,0.5]]' \
  -F 'bounding_box_labels=[1]' \
  -F response_format=aspire
```

### Responses

All formats include `image_size: [height, width]` and `outputs` with legacy parallel arrays.

`response_format=legacy` (default):

```json
{
  "image_size": [480, 640],
  "outputs": {
    "object_ids": [0, 1],
    "probabilities": [0.95, 0.88],
    "boxes_xywh": [[0.1, 0.1, 0.2, 0.2], [0.5, 0.5, 0.2, 0.2]],
    "mask_count": 2,
    "masks_rle": [{"size": [480, 640], "counts": "..."}]
  }
}
```

`response_format=aspire` adds `prompt` and `instances`:

```json
{
  "prompt": "bowl",
  "image_size": [480, 640],
  "instances": [
    {
      "object_id": 0,
      "score": 0.95,
      "label": "bowl",
      "box_xywh": [0.1, 0.1, 0.5, 0.5],
      "mask_rle": {"size": [480, 640], "counts": "..."}
    }
  ],
  "outputs": { "...": "same legacy arrays as above" }
}
```

Use `box_format=xywh_pixel` with `response_format=aspire` when clients need pixel-space boxes. Decode masks with `pycocotools.mask.decode` and cast to `uint8` for downstream geometry.
