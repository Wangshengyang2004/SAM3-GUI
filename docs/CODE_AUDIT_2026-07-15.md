# Code Audit - 2026-07-15

This audit covered the FastAPI surface, Gradio session lifecycle, SAM 3.1 backend concurrency, image/video state transitions, destructive filesystem operations, packaging, dependencies, and tests.

## Fixed

- Isolated mutable Gradio handlers by browser session while sharing the model backend.
- Serialized predictor initialization and inference/session state transitions.
- Confined API resource paths to the configured data root and removed wildcard CORS.
- Added request shape/range validation, upload byte/pixel/format limits, and propagation limits.
- Stopped API health and internal errors from exposing server paths or exception details.
- Made video frame extraction transactional and prevented path traversal or destructive failure.
- Replayed video prompt state after point/object edits and invalidated stale propagation output.
- Made image switching transactional and saved all mask instances atomically.
- Made tracked-video outputs unique and atomically published.
- Moved the application into an installable package with Hydra resources in wheel/sdist.
- Upgraded the Web stack and added dependency, lint, build, and offline-test CI gates.

## Verified

- Offline API, UI isolation, backend, handler, CLI, and filesystem tests pass on Python 3.12.
- Wheel and sdist build successfully; the wheel works outside the source tree.
- Ruff and Bandit report no high-severity findings.

## Residual Risks

- Real SAM 3.1 inference tests require a local checkout, checkpoint, CUDA environment, and media fixtures. They are marked `integration` and are not run by hosted CPU CI.
- The API has no application-level authentication. The default bind address is localhost; remote deployments must explicitly opt in and enforce trusted-network or reverse-proxy authentication.
- Long video propagation still buffers response data and may require job-based streaming for untrusted, high-volume deployments.
- The repository has no explicit license file or package license metadata. The owner must choose a license compatible with the upstream SAM3 code and model terms before public redistribution.
- Browser-level Playwright coverage and fixed-fixture segmentation quality thresholds remain future hardening work.
