# DepthLens Pro runtime models directory

This directory is packaged into the native Electron app as a runtime resource.
It must exist in source checkouts and in built app artifacts even when no model
weights are committed here.

Large model binaries and generated ONNX files are intentionally not committed by
default. Generate or download runtime assets through the documented setup/export
commands when they are explicitly needed.
