import cv2
import torch
from torchvision.transforms import Compose, Resize, ToTensor
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def estimate_depth(image_path):
    # Load MiDaS model (only when function is called)
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").eval()
    transform = Compose([Resize((384, 384)), ToTensor()])
    
    # Load image with OpenCV
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Convert BGR to RGB and numpy array to PIL Image
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    
    # Apply transforms
    input_tensor = transform(pil_image).unsqueeze(0)
    
    with torch.no_grad():
        depth = midas(input_tensor)
    depth = depth.squeeze().cpu().numpy()
    return depth

# Test code (only runs if this file is executed directly)
if __name__ == "__main__":
    print('start', flush=True)
    depth_map = estimate_depth('output/frames/frame_0000.png')
    print(depth_map, flush=True)
    print('done', flush=True)