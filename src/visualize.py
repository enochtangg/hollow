import os
import cv2
import torch
import numpy as np
# import open3d as o3d
from tqdm import tqdm
from torchvision.transforms import Compose, Resize, ToTensor

def load_midas():
    model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    model.eval()
    transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
    return model, transform

def estimate_depth(image, model, transform):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_tensor = transform(image_rgb).unsqueeze(0)
    with torch.no_grad():
        prediction = model(input_tensor)
    return prediction.squeeze().cpu().numpy()

def frame_to_point_cloud(depth_map, image_rgb):
    h, w = depth_map.shape
    fx = fy = 1
    cx, cy = w / 2, h / 2

    points = []
    colors = []

    for v in range(0, h, 2):  # subsample for speed
        for u in range(0, w, 2):
            Z = depth_map[v, u]
            if Z <= 0 or Z > 10:  # discard far or invalid points
                continue
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            points.append([X, Y, Z])
            colors.append(image_rgb[v, u] / 255.0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def load_frames(directory):
    return sorted([
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(".png")
    ])

def main(frames_dir):
    print(f"📂 Loading frames from: {frames_dir}")
    frame_paths = load_frames(frames_dir)
    if not frame_paths:
        print("❌ No .png frames found.")
        return

    print("📦 Loading MiDaS model")
    model, transform = load_midas()

    all_pcds = []
    for path in tqdm(frame_paths, desc="🔍 Processing frames"):
        image = cv2.imread(path)
        depth = estimate_depth(image, model, transform)
        pcd = frame_to_point_cloud(depth, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        all_pcds.append(pcd)

    print(all_pcds)

    print("🧠 Merging point clouds...")
    combined_pcd = all_pcds[0]
    for p in all_pcds[1:]:
        combined_pcd += p
    combined_pcd = combined_pcd.voxel_down_sample(voxel_size=0.01)
    combined_pcd.estimate_normals()

    print("🖼️ Rendering final 3D point cloud")
    o3d.visualization.draw_geometries([combined_pcd])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Combine video frames into a 3D point cloud")
    parser.add_argument("frames_dir", help="Directory containing PNG frames")
    args = parser.parse_args()

    main(args.frames_dir)