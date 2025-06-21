import os
import sys

import open3d as o3d
from depth import estimate_depth

print("1. Starting test...", flush=True)

# Test frame extraction
print("2. Testing frame extraction...", flush=True)
from frames import extract_frames

video_path = "sample_videos/room.mp4"
output_frames = "output/frames"

print(f"3. Video path: {video_path}", flush=True)
print(f"4. Video exists: {os.path.exists(video_path)}", flush=True)

try:
    extract_frames(video_path, output_frames)
    print("5. Frame extraction completed", flush=True)
except Exception as e:
    print(f"5. Frame extraction failed: {e}", flush=True)
    sys.exit(1)

# Check if frames were created
if os.path.exists(output_frames):
    frame_files = sorted([f for f in os.listdir(output_frames) if f.endswith(".png")])
    print(f"6. Found {len(frame_files)} frames", flush=True)
    
    if frame_files:
        frame_path = os.path.join(output_frames, frame_files[0])
        print(f"7. First frame path: {frame_path}", flush=True)
        print(f"8. Frame exists: {os.path.exists(frame_path)}", flush=True)
        
        # Test depth estimation
        print("9. Testing depth estimation...", flush=True)
        try:
            
            print("10. Depth module imported successfully", flush=True)
            print(frame_path, flush=True)
            depth_map = estimate_depth(frame_path)
            print(f"11. Depth estimation completed. Shape: {depth_map.shape}", flush=True)
            
            # Test point cloud generation
            print("12. Testing point cloud generation...", flush=True)
            try:
                from point_cloud import create_point_cloud
                print("13. Point cloud module imported successfully", flush=True)
                
                pcd = create_point_cloud(depth_map, frame_path)
                print(f"14. Point cloud created successfully. Points: {len(pcd.points)}", flush=True)
                
                # Test mesh generation
                print("15. Testing mesh generation...", flush=True)
                try:
                    from mesh import generate_mesh
                    print("16. Mesh module imported successfully", flush=True)
                    
                    mesh = generate_mesh(pcd)
                    print(f"17. Mesh generated successfully. Vertices: {len(mesh.vertices)}, Triangles: {len(mesh.triangles)}", flush=True)
                    # o3d.visualization.draw_geometries([mesh])
                    
                except Exception as e:
                    print(f"17. Mesh generation failed: {e}", flush=True)
                
            except Exception as e:
                print(f"14. Point cloud generation failed: {e}", flush=True)
                sys.exit(1)
            
        except Exception as e:
            print(f"11. Depth estimation failed: {e}", flush=True)
            sys.exit(1)
    else:
        print("6. No frames found", flush=True)
else:
    print("6. Output directory not created", flush=True)

print("18. Test completed successfully!", flush=True) 