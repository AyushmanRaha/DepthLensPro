# DepthLensPro
**AI-Powered 2D to 3D Depth Map Generation Web Application**

DepthLensPro is an intelligent local web application that converts 2D images into high-quality depth visualizations using monocular depth estimation (MiDaS family). It combines a FastAPI inference backend (v3.1.0) with a production-style browser workspace for batch processing, device-aware execution, model comparison, and runtime analytics.

---

## Features

* AI-powered monocular depth estimation with multiple MiDaS models (`MiDaS_small`, `DPT_Hybrid`, `DPT_Large`)
* Web-based interface (runs locally with backend + static frontend)
* Batch processing support (up to 10 images per request)
* Broad image format compatibility (PNG, JPG/JPEG, WEBP, BMP)
* Dual output rendering (colorized depth map + grayscale depth map)
* Rich colormap support (`inferno`, `plasma`, `viridis`, `magma`, `jet`, `hot`, `bone`, `turbo`)
* Compute-device routing with `Auto` and explicit `CPU`, `CUDA`, `MPS`, and `XPU` selection where available
* Expanded MDE metric suite (SSIM, SILog, PSNR, entropy, dynamic range, edge/gradient statistics, histogram spread, and consistency proxies)
* Dedicated model comparison panel with side-by-side output cards and metric-driven charting
* Session analytics dashboard (latency trends, throughput, cache hits, error counts, total inference time)
* In-memory inference caching for repeated image + model + colormap + device combinations
* API-first backend architecture for integration, automation, and deployment workflows

---

## What's New (Latest Codebase Updates)

* Backend upgraded to **v3.1.0**.
* Device detection and prioritization logic refined:
  * corrected compute-class semantics for Apple MPS and Intel XPU,
  * default device priority now `CUDA > MPS > XPU > CPU`,
  * MPS runtime checks now enforce both build-time and runtime availability guards.
* Backend startup lifecycle improved with warm-up on best available accelerator instead of CPU-only initialization.
* Runtime acceleration checks in `/health` now reflect only operational PyTorch backends.
* Request safety and reliability remain enforced through:
  * per-file size cap (**20 MB**),
  * image dimension normalization cap (**2048 px max side**),
  * batch upper limit (**10 images**),
  * standardized JSON error responses.
* Frontend experience includes:
  * persistent user preferences (model, colormap, device),
  * improved upload flow with cancel support and ETA/progress visibility,
  * dynamic device inventory rendering from backend capability APIs,
  * enhanced compare workflow with selectable benchmark metrics,
  * lightbox-level metric surfacing and download actions.

---

## Project Structure

```text
DepthLensPro/
│
├── backend/                  # FastAPI + depth inference engine
│   ├── app.py                # Main API server (v3.1.0)
│   ├── depth_models.py       # Auxiliary depth model loader class
│   └── requirements.txt      # Python dependencies
│
├── frontend/                 # Web UI
│   ├── index.html            # Multi-panel UI (Workspace / Compare / About)
│   ├── style.css             # Cyber-neon visual design system
│   └── script.js             # Frontend logic (state, API, charts, metrics)
│
└── README.md
```

---

## Tech Stack

* **Backend:** Python, FastAPI, Uvicorn
* **AI Models:** MiDaS (`MiDaS_small`, `DPT_Hybrid`, `DPT_Large`) via PyTorch Hub
* **Frontend:** HTML, CSS, JavaScript
* **Core Libraries:** PyTorch, OpenCV, NumPy, Chart.js
* **Transport/Data:** Multipart uploads + JSON API responses + base64 output payloads

---

## Installation & Setup

### Prerequisites

Make sure you have installed:

* Python 3.10+
* pip
* Git

> Note: First-time model loading downloads weights from `intel-isl/MiDaS` via `torch.hub`, so internet access is required during the initial model bootstrap.

---

## Setup Instructions (OS-Specific)

### macOS

```bash
# Clone repository
git clone https://github.com/AyushmanRaha/DepthLensPro.git
cd DepthLensPro

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt

# Run backend server
uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

### Windows

```bash
# Clone repository
git clone https://github.com/AyushmanRaha/DepthLensPro.git
cd DepthLensPro

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r backend/requirements.txt

# Run backend server
uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

### Linux

```bash
# Clone repository
git clone https://github.com/AyushmanRaha/DepthLensPro.git
cd DepthLensPro

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt

# Run backend server
uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

---

## Running the Application

**1. Start the backend server:**

```bash
uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

**2. Open frontend:**

Serve the frontend directory via a local HTTP server:

```bash
cd frontend
python -m http.server 5500
```

**3. Visit:**
`http://localhost:5500`

**4. Verify backend health (optional):**

```bash
curl http://127.0.0.1:8000/health
```

---

## Workflow

### Workspace Flow

1. Upload one or more 2D images to the queue.
2. Select model architecture, colormap, and compute device strategy.
3. Submit request to backend for depth inference.
4. Review output cards (original, depth map, grayscale depth) and metric panel.
5. Re-run equivalent inputs/settings to benefit from in-memory cache acceleration.

### Compare Flow

1. Switch to the **COMPARE** panel.
2. Upload a single image for controlled benchmark comparison.
3. Execute all supported models against the same source image.
4. Analyze side-by-side outputs, latency behavior, and selected quality metric trends.

---

## API Overview

Base URL: `http://127.0.0.1:8000`

* `GET /` — API metadata + service version
* `GET /health` — service/runtime health (devices, acceleration checks, cache/system metadata)
* `GET /devices` — discover available compute targets and capabilities
* `GET /models` — supported model registry and performance notes
* `GET /colormaps` — supported colormap keys
* `POST /estimate` — single-image depth estimation
* `POST /batch` — multi-image depth estimation (max 10 files)
* `DELETE /cache` — clear in-memory inference cache

---

## Example Output

* **Input Image** -> RGB source image
* **Output 1** -> Depth map (selected colormap)
* **Output 2** -> Grayscale normalized depth map
* **Metadata** -> Model name, resolved compute device, latency, resolution, cache status, and complete metric bundle

---

## Future Enhancements

* Video depth estimation pipeline
* True 3D reconstruction (point clouds / meshes)
* Persistent cache layer (Redis/disk)
* Dockerized deployment workflow
* Mobile-responsive UX improvements

---

## Contribution

Contributions are welcome.

1. Fork the repository
2. Create a branch: `git checkout -b feature/your-feature-name`
3. Commit changes: `git commit -m "Describe your change"`
4. Push branch and open a Pull Request

---

## Troubleshooting

| Issue | Solution |
| :--- | :--- |
| `ModuleNotFoundError` | Ensure virtual environment is activated and dependencies are installed |
| Frontend shows "Engine offline" | Confirm FastAPI is running on `127.0.0.1:8000` |
| Browser CORS/network issue | Open frontend through local HTTP server (`python -m http.server`) instead of `file://` |
| Slow inference | Prefer `MiDaS_small` or choose `CUDA`/`MPS`/`XPU` where available |
| Port already in use | Change port using `--port` in uvicorn command |
| First run is slow | MiDaS model weights are downloaded once and cached locally |

---

## License

This project is licensed under the MIT License.

---

## Authors

**Primary Author and Maintainer: Ayushman Raha**
GitHub: [https://github.com/AyushmanRaha](https://github.com/AyushmanRaha)

---

## Acknowledgements

* MiDaS Depth Estimation Models (Intel ISL)
* OpenCV & PyTorch communities
* FastAPI ecosystem

---

## Notes

* This project is optimized for local development.
* Inference cache is in-memory and resets when the backend restarts.
* Maximum supported image size is 20 MB and max dimension is 2048 px (auto-resized above this).
* GPU acceleration depends on local PyTorch backend support (`CUDA`, `MPS`, or `XPU`) and installed drivers/runtime.
