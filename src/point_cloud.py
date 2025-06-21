import cv2
import numpy as np
import open3d as o3d

def create_point_cloud(depth_map, rgb_image_path):
    Load RGB image
    rgb = cv2.imread(rgb_image_path)
    if rgb is None:
        raise ValueError(f"Could not read image at {rgb_image_path}")

    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    Resize to match depth map
    if rgb.shape[:2] != depth_map.shape:
        print("⚠️ Resizing RGB to match depth map")
        rgb = cv2.resize(rgb, (depth_map.shape[1], depth_map.shape[0]))

    h, w = depth_map.shape
    fx = fy = 1  # Focal length approximation
    cx, cy = w / 2, h / 2

    points = []
    colors = []

    for v in range(0, h, 2):  # Subsample for performance
        for u in range(0, w, 2):
            Z = depth_map[v, u]
            if Z <= 0 or Z > 10:
                continue
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            points.append([X, Y, Z])
            colors.append(rgb[v, u] / 255.0)

    if len(points) == 0:
        raise ValueError("No valid depth points found — check depth map output.")

    # Build Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))

    return pcd