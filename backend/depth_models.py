import torch
import cv2

class DepthEstimator:
    def __init__(self):
        # Prefer Metal (MPS) on Mac for speed, fallback to CPU
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.models = {}
        self.transforms = {}
        self.repo = "intel-isl/MiDaS"

    def load_model(self, model_name):
        if model_name not in self.models:
            print(f"Loading {model_name} onto {self.device}...")
            self.models[model_name] = torch.hub.load(self.repo, model_name)
            self.models[model_name].to(self.device)
            self.models[model_name].eval()
            
            midas_transforms = torch.hub.load(self.repo, "transforms")
            if model_name == "MiDaS_small":
                self.transforms[model_name] = midas_transforms.small_transform
            else:
                self.transforms[model_name] = midas_transforms.dpt_transform
                
        return self.models[model_name], self.transforms[model_name]

    def predict(self, img_rgb, model_name):
        model, transform = self.load_model(model_name)
        input_batch = transform(img_rgb).to(self.device)

        with torch.no_grad():
            prediction = model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        return prediction.cpu().numpy()