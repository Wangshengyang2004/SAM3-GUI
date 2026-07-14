# Repository Guidelines

## Project Structure & Module Organization
- `sam3_gui/`: application package containing the Gradio UI, FastAPI routes, SAM 3.1 backend, handlers, config loader, and utilities.
- `cli.py`: root launcher; accepts Hydra overrides such as `server.port=8891 sam.use_fa3=true`.
- `sam3_gui/conf/`: packaged Hydra configuration; `config.yaml` composes `server`, `data`, and `sam` groups.
- `tests/`: pytest suite for model loading, prompt flow, propagation, and mask utilities (`test_*.py`).
- `data_root/`: sample input/output layout (`videos/`, `images/`, `masks/`) used for local runs and test fixtures.
- `docs/`: secondary documentation, localized README, changelog, hardware notes, and screenshot assets.
- `runtime/`: ignored local logs and generated outputs; treat as working data, not source code.

## Build, Test, and Development Commands
- Install dependencies:
  - `pip install -r requirements.txt`
  - Install SAM3 with SAM 3.1 support first (required): `pip install -e /path/to/sam3`
- Run the app locally:
  - `python cli.py data.root_dir=data_root server.port=8890`
- Run tests:
  - `pytest -q`
  - Single file: `pytest tests/test_mask_utils.py -q`

## Coding Style & Naming Conventions
- Follow Python conventions: 4-space indentation, `snake_case` for functions/variables, `PascalCase` for classes.
- Keep functions small and state transitions explicit (especially around session reset, frame index, and mask dictionaries).
- Prefer clear, side-effect-aware method names (`set_img_dir`, `add_text_prompt`, `clear_points` style).
- Avoid new hardcoded absolute paths; use CLI args or environment variables.

## Testing Guidelines
- Framework: `pytest` with shared fixtures in `tests/conftest.py`.
- Name tests as `test_*` and group related behavior in `Test*` classes.
- Integration tests depend on SAM 3.1 and may use CUDA if available.
- Set these environment variables when needed:
  - `SAM3_CHECKPOINT_PATH` for model checkpoint location
  - `SAM3_TEST_IMG_DIR` for fixture image directory

## Commit & Pull Request Guidelines
- Existing history uses short, one-line subjects (for example, “Update requirements.txt”, “fixing bugs”).
- Keep commit titles imperative and specific; prefer `area: change` style (example: `tests: add text prompt edge case`).
- PRs should include:
  - What changed and why
  - How to validate (`pytest -q`, manual GUI flow)
  - Screenshots/GIFs for UI behavior changes
  - Notes on SAM 3.1/model/data assumptions
