import torch
from segment_anything import sam_model_registry, SamPredictor
import cv2

def segment_frame(image_path):
    # Load SAM model (only when function is called)
    sam = sam_model_registry["vit_h"](checkpoint="models/sam_vit_h_4b8939.pth")
    predictor = SamPredictor(sam)
    
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)
    masks, _, _ = predictor.predict(point_coords=None, point_labels=None)
    return masks
