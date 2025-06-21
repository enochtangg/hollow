import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor

# Load the image
image_path = "output/frames/frame_0000.png"
image_bgr = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# Load the SAM model
checkpoint_path = "models/sam_vit_h_4b8939.pth"
sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
predictor = SamPredictor(sam)

# Run prediction
predictor.set_image(image_rgb)
masks, scores, _ = predictor.predict(point_coords=None, point_labels=None, multimask_output=True)

# Visualize the masks over the image
def show_masks(image, masks):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for i in range(masks.shape[0]):
        plt.imshow(masks[i], alpha=0.4)
    plt.title("Segmented Masks")
    plt.axis('off')
    plt.show()

print(f"Number of masks: {masks.shape[0]}")
show_masks(image_rgb, masks)