import time
import cv2
import base64
import numpy as np
from typing import List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from depth_models import DepthEstimator

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

estimator = DepthEstimator()

@app.post("/predict")
async def predict_multiple_images(
    files: List[UploadFile] = File(...), 
    model_name: str = Form(...)
):
    if len(files) > 5:
        raise HTTPException(status_code=400, detail="Maximum 5 files allowed.")
    
    results = []
    
    for file in files:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            continue
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        start_time = time.time()
        try:
            depth_map = estimator.predict(img_rgb, model_name)
        except Exception as e:
            print(f"Prediction error for {file.filename}: {e}")
            continue
            
        inference_time = time.time() - start_time
        
        # Normalize and colorize depth map
        depth_min, depth_max = depth_map.min(), depth_map.max()
        depth_scaled = (255 * (depth_map - depth_min) / (depth_max - depth_min)).astype("uint8")
        depth_colored = cv2.applyColorMap(depth_scaled, cv2.COLORMAP_INFERNO)
        
        # Base64 Encode
        _, buffer = cv2.imencode('.png', depth_colored)
        depth_base64 = base64.b64encode(buffer).decode('utf-8')
        
        results.append({
            "filename": file.filename,
            "depth_map": depth_base64,
            "inference_time": round(inference_time, 3),
            "resolution": f"{img.shape[1]}x{img.shape[0]}"
        })
        
    return {
        "model_used": model_name,
        "results": results
    }