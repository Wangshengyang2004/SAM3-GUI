# SAM3.1 GUI/API Status

This project now targets only native SAM 3.1 Object Multiplex.

## Implemented

- Gradio UI mounted through FastAPI.
- `/api/health`, `/api/sessions`, `/api/prompts`, `/api/propagate`, `/api/objects/remove`, and `/api/images/segment`.
- Video text prompts, point prompts, box prompts, object removal, propagation direction, tracked video preview, and mask export.
- Image text, point, and box prompts through a single-frame SAM 3.1 session.
- SAM 3.1 checkpoint downloader for `facebook/sam3.1/sam3.1_multiplex.pt` from Hugging Face or ModelScope.
- Real `.mp4` integration coverage with text+box prompting and propagation.
- Real online image inference smoke checks for common open-vocabulary concepts.

## Notes

- SAM 3.1 Object Multiplex may not create new objects from pure box-only prompts. Prefer text prompts, optionally with boxes as geometric guidance.
- Empty prompt results intentionally skip propagation to avoid native empty-object failures.

## Follow-Up Candidates

- Add streaming server-sent events for long `/api/propagate` calls.
- Add API examples for mask RLE decoding in Python and JavaScript.
- Add optional batch image endpoint when the UI needs it.
