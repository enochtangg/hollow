import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, Resize, ToTensor

# Load MiDaS model (you can swap this with your custom loader)
def load_midas_model():
    model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    model.eval()
    transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
    return model, transform

# Estimate depth
def estimate_depth(image_path, model, transform):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        prediction = model(input_tensor)
        depth = prediction.squeeze().cpu().numpy()
    return depth

# Test function
def test_depth_estimation():
    test_image = "output/frames/frame_0000.png"
    assert os.path.exists(test_image), f"Test image not found at {test_image}"

    model, transform = load_midas_model()
    depth_map = estimate_depth(test_image, model, transform)

    # Basic assertions
    assert isinstance(depth_map, np.ndarray), "Depth map must be a NumPy array"
    assert len(depth_map.shape) == 2, "Depth map must be a 2D array"
    assert np.count_nonzero(depth_map) > 0, "Depth map must contain non-zero values"
    assert np.all(depth_map >= 0), "Depth values must be non-negative"

    print("✅ Depth estimation test passed.")
    return depth_map

# Optional: Visualize
if __name__ == "__main__":
    depth = test_depth_estimation()
    plt.imshow(depth, cmap='inferno')
    plt.colorbar()
    plt.title("Estimated Depth Map")
    plt.axis('off')
    plt.show()