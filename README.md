# DepthLensPro
**AI-Powered 2D to 3D Depth Map Generation Web Application**

DepthLensPro is an intelligent web-based system that transforms 2D images into high-quality depth visualizations using state-of-the-art monocular depth estimation (MiDaS family). It includes a FastAPI backend and a polished browser UI for batch inference, model/device selection, result exploration, model comparison, and runtime analytics.

---

## Features

* AI-powered monocular depth estimation with multiple MiDaS models
* Web-based interface (runs locally)
* Batch processing (up to 10 images per request)
* Supports common image formats (PNG, JPG, WEBP, BMP)
* Depth visualization (grayscale + selectable colormaps)
* Compute-device routing with Auto / CPU / CUDA / Apple Silicon (MPS)
* Rich MDE metric suite (distribution, edge, structural, and quality proxies)
* Compare panel for side-by-side model performance on a single image
* Session analytics dashboard (latency, throughput, cache hits, errors)
* In-memory inference result caching for repeated inputs
* API-ready backend for extension and deployment

---

## What's New (Latest Codebase Updates)

* Backend upgraded to **v3.0.0** with improved device detection and health telemetry.
* Added explicit device APIs and runtime hardware introspection (`/health`, `/devices`).
* Added expanded depth-quality metric bundle (SSIM, entropy, dynamic range, gradient/edge stats, SILog, PSNR, histogram, and more).
* Added stronger request guards: size limit (20 MB), max dimension resize (2048 px), per-batch limit (10 images), and structured error handling.
* Frontend upgraded to **v3.0** with:
  * persistent model/colormap/device preferences,
  * improved upload flow (browse reliability, cancel support, ETA/progress tracking),
  * dynamic device selector and engine status indicators,
  * model comparison workflow with latency charting,
  * expanded per-session dashboard + lightbox metric surfacing.

---

## Project Structure

```text
DepthLensPro/
│
├── backend/                  # FastAPI + depth inference engine
│   ├── app.py                # Main API server (v3.0.0)
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
* **AI Models:** MiDaS (`MiDaS_small`, `DPT_Hybrid`, `DPT_Large`)
* **Frontend:** HTML, CSS, JavaScript
* **Libraries:** PyTorch, OpenCV, NumPy, Chart.js

---

## Installation & Setup

### Prerequisites

Make sure you have installed:

* Python 3.10+
* pip
* Git

> Note: First-time model loading downloads weights from `intel-isl/MiDaS` via `torch.hub`, so internet access is required during initial run.

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

1. Upload one or more 2D images.
2. Select model, colormap, and compute device.
3. Backend processes images and generates normalized depth maps.
4. UI displays depth output, grayscale output, and quality/latency metrics.
5. Re-running identical input/settings returns cached results faster.

### Compare Flow

1. Switch to **COMPARE** panel.
2. Upload a single image for benchmark comparison.
3. Run all supported models against the same input.
4. Review side-by-side outputs and latency chart for model trade-offs.

---

## API Overview

Base URL: `http://127.0.0.1:8000`

* `GET /` — API metadata and version
* `GET /health` — Service/device/model/cache/system health snapshot
* `GET /devices` — Available compute devices (CPU/CUDA/MPS)
* `GET /models` — Supported model list + performance notes
* `GET /colormaps` — Supported colormap names
* `POST /estimate` — Single-image depth estimation
* `POST /batch` — Batch depth estimation (max 10 images)
* `DELETE /cache` — Clear in-memory inference cache

---

## Example Output

* **Input Image** -> RGB Image
* **Output 1** -> Depth Map (selected colormap)
* **Output 2** -> Grayscale Depth Map
* **Metadata** -> Inference latency, resolution, device used, and MDE metric bundle

---

## Future Enhancements

* Video depth estimation pipeline
* True 3D reconstruction (point clouds / meshes)
* Persistent cache layer (Redis/disk)
* Dockerized deployment workflow
* Mobile-responsive UX improvements

---

## Contribution

Contributions are welcome!

1. Fork the repo
2. Create a new branch: `git checkout -b feature/your-feature-name`
3. Commit changes: `git commit -m "Add new feature"`
4. Push and create PR: `git push origin feature/your-feature-name`

---

## Troubleshooting

| Issue | Solution |
| :--- | :--- |
| `ModuleNotFoundError` | Ensure virtual environment is activated and dependencies are installed |
| Frontend shows "Engine offline" | Confirm FastAPI is running on `127.0.0.1:8000` |
| CORS/network issues in browser | Open frontend via local server (`python -m http.server`) instead of `file://` |
| Slow inference | Use `MiDaS_small` or select CUDA/MPS device if available |
| Port already in use | Change with `--port` in uvicorn command |
| First run is slow | Model weights are downloaded once and then cached locally |

---

## License

This project is licensed under the MIT License.

---

## Authors

**Primary Author and Maintainer: Ayushman Raha** GitHub: [https://github.com/AyushmanRaha](https://github.com/AyushmanRaha)

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
* GPU support depends on your local PyTorch/CUDA or Apple Metal (MPS) setup.
