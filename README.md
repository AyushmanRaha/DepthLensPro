# DepthLensPro
**AI-Powered 2D to 3D Depth Map Generation Web Application**

DepthLensPro is an intelligent web-based system that transforms 2D images into high-quality 3D depth maps using state-of-the-art deep learning models. Designed as a fast, modular, and scalable project, it enables users to upload images and visualize depth information seamlessly in a browser.

---

## Features

* AI-powered monocular depth estimation
* Web-based interface (runs on localhost)
* Fast image processing pipeline
* Supports common image formats (JPG, PNG)
* Depth visualization (grayscale / colormap)
* Modular architecture for easy extension
* API-ready backend for future scalability

---

## Project Structure

```text
DepthLensPro/
│
├── backend/           # Core AI + server logic
│   ├── models/        # Pretrained depth models
│   ├── utils/         # Helper functions
│   ├── app.py         # Main backend server
│   └── requirements.txt
│
├── frontend/          # Web UI
│   ├── index.html
│   ├── styles.css
│   └── script.js
│
├── assets/            # Sample images / outputs
├── README.md
└── .gitignore
```

---

## Tech Stack

* **Backend:** Python, Flask / FastAPI  
* **AI Models:** MiDaS / Depth Estimation Models  
* **Frontend:** HTML, CSS, JavaScript  
* **Libraries:** OpenCV, PyTorch, NumPy  

---

## Installation & Setup

### Prerequisites

Make sure you have installed:

* Python 3.8+
* pip
* Git

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
python backend/app.py
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
python backend/app.py
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
python3 backend/app.py
```

---

## Running the Application

**1. Start the backend server:**

```bash
python backend/app.py
```

**2. Open frontend:**

Open `frontend/index.html` in your browser OR serve via a local server:

```bash
cd frontend
python -m http.server 5500
```

**3. Visit:**
`http://localhost:5500`

---

## Workflow

1.  Upload a 2D image
2.  Backend processes image using AI model
3.  Depth map is generated
4.  Result is displayed in browser

---

## Example Output

* **Input Image** -> RGB Image
* **Output** -> Depth Map (Grayscale / Heatmap)

---

## Future Enhancements

* Video depth estimation
* Full 3D reconstruction (point clouds)
* Cloud deployment (AWS / GCP)
* GPU acceleration support
* Mobile-friendly UI

---

## Contribution

Contributions are welcome!

1.  Fork the repo
2.  Create a new branch: `git checkout -b feature/your-feature-name`
3.  Commit changes: `git commit -m "Add new feature"`
4.  Push and create PR: `git push origin feature/your-feature-name`

---

## Troubleshooting

| Issue | Solution |
| :--- | :--- |
| Module not found | Ensure virtual environment is activated |
| Slow performance | Use GPU (if supported) |
| Port already in use | Change port in backend config |

---

## License

This project is licensed under the MIT License.

---

## Author

**Ayushman Raha** GitHub: [https://github.com/AyushmanRaha](https://github.com/AyushmanRaha)

---

## Acknowledgements

* MiDaS Depth Estimation Models
* OpenCV & PyTorch communities

---

## Notes

* This project is optimized for local development
* Ensure proper dependencies are installed before running
* GPU support requires additional configuration
