# DepthLens Pro: Advanced Monocular Depth Estimation

## Overview
DepthLens Pro is a comprehensive, full-stack machine vision application that transforms 2D images into detailed 3D depth maps. It provides a professional dashboard for comparing three distinct MiDaS model architectures (Small, Hybrid, and Large) to analyze the trade-off between inference speed and accuracy. 

The application supports simultaneous processing of multiple images and features a highly interactive, animated, and responsive frontend with real-time performance tracking.

## Architecture
* **Frontend:** A modern, single-page application built with HTML, CSS, and JavaScript. Features asynchronous multi-file handling, a sophisticated dark-themed dashboard, CSS animations, and Chart.js for data visualization.
* **Backend:** A FastAPI server built with Python. Handles multi-file CORS requests, image decoding via OpenCV, and model inference.
* **Machine Learning:** PyTorch implementation of MiDaS via Torch Hub.
    * `MiDaS_small`: High speed, lightweight.
    * `DPT_Hybrid`: Balanced performance.
    * `DPT_Large`: High accuracy, high computational cost.

## Installation and Setup

### 1. Backend Setup
1.  Navigate to the backend directory:
    ```bash
    cd ~/Downloads/DepthLensPro/backend
    ```
2.  Install dependencies:
    ```bash
    pip3 install -r requirements.txt
    ```
3.  Start the server:
    ```bash
    uvicorn app:app --reload --host 127.0.0.1 --port 8000
    ```
    *Note: The first run will automatically download model weights from Torch Hub.*

### 2. Frontend Setup
1.  Navigate to the frontend directory:
    ```bash
    cd ~/Downloads/DepthLensPro/frontend
    ```
2.  Start a local HTTP server:
    ```bash
    python3 -m http.server 3000
    ```

### Accessing the Application
Once both servers are running, open Google Chrome and navigate to:
`http://localhost:3000`

## System Usage
1.  Select a model architecture from the control panel.
2.  Click "Browse Files" or drag & drop to upload images.
3.  Review the file queue.
4.  Click "Generate Depth Maps". The progress bar provides real-time feedback.
5.  View the generated depth maps in the results gallery and track performance in the dashboard.