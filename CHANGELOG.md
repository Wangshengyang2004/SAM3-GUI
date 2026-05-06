# Changelog

## Unreleased

- Retargeted the app to native SAM 3.1 Object Multiplex with `sam3.1_multiplex.pt`.
- Added FastAPI endpoints for health, sessions, prompts, propagation, object removal, and single-image segmentation.
- Mounted the existing Gradio GUI under the FastAPI app at `/`.
- Added checkpoint download support for both Hugging Face and ModelScope (`facebook/sam3.1`).
- Added SAM 3.1 backend validation, checkpoint status reporting, and COCO RLE API serialization.
- Added real-inference integration coverage for image sessions, frame-directory sessions, and `.mp4` propagation.
- Documented the SAM 3.1 prompt behavior: use text prompts, optionally with boxes, when creating new video objects.
- Added Blackwell/RTX 5090 torchvision build notes.
