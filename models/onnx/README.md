# Optional ONNX runtime assets

This directory must exist in every packaged DepthLens Pro app so startup checks,
ONNX diagnostics, and PyTorch fallback paths can resolve a stable runtime
location.

The `.onnx` files themselves are optional for the default native build. Exported
ONNX graphs such as `midas_small.onnx`, `dpt_hybrid.onnx`, and `dpt_large.onnx`
are generated separately and should not be committed unless a release process
explicitly intends to include those binaries.
