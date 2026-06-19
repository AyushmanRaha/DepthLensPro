# User Guide

This guide preserves the extended feature tour and screenshot references from the main README.

## Workspace

Generate depth maps from uploaded images, choose model/device/colormap settings, inspect output, and export results.

<p align="center"><img src="screenshots/workspace-generate-depth-maps.png" alt="Workspace — Generate Depth Maps" width="900"></p>

## Ground Truth

Upload GT depth alongside an image to compute benchmark-style metrics and compare prediction quality.

<p align="center"><img src="screenshots/ground-truth-mode.png" alt="Ground Truth Mode" width="900"></p>

## Webcam

Run live local webcam depth streaming when camera access and local runtime conditions are available.

<p align="center"><img src="screenshots/webcam-live-depth-streaming.png" alt="Webcam — Live Depth Streaming" width="900"></p>

## Compare

Run all supported models on one image and compare latency, output appearance, and diagnostics side by side.

<p align="center"><img src="screenshots/compare-run-all-models.png" alt="Compare — Run All Models on One Image" width="900"></p>

## Performance

Use the performance view for PyTorch versus optional ONNX Runtime benchmark comparisons.

<p align="center"><img src="screenshots/performance-pytorch-vs-onnx.png" alt="Performance — PyTorch vs ONNX Runtime" width="900"></p>

## Experiments

Create reproducible validation runs and preserve run metadata for reviewer-friendly comparisons.

<p align="center"><img src="screenshots/experiments-validation-runs.png" alt="Experiments — Reproducible Validation Runs" width="900"></p>

## 3D Reconstruction

Generate approximate point-cloud previews and export PLY/OBJ data from predicted depth.

<p align="center"><img src="screenshots/3d-reconstruction.png" alt="3D Reconstruction" width="900"></p>

## Guide

Use the offline in-app guide as a local reference for workflow and troubleshooting.

<p align="center"><img src="screenshots/guide-offline-reference.png" alt="Guide — Offline In-App Reference" width="900"></p>

## Observability

DepthLens Pro exposes local observability through health/readiness routes, cache metrics, Prometheus metrics, and the app's local runtime telemetry views. These diagnostics are designed to avoid raw image uploads, cloud telemetry, filenames, local paths, and high-cardinality image identifiers.
