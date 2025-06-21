import cv2
import os

def extract_frames(video_path, output_dir, fps=2):
    os.makedirs(output_dir, exist_ok=True)
    vidcap = cv2.VideoCapture(video_path)
    count, frame_idx = 0, 0
    frame_rate = vidcap.get(cv2.CAP_PROP_FPS)
    
    while True:
        success, image = vidcap.read()
        if not success:
            break
        if int(count % (frame_rate // fps)) == 0:
            cv2.imwrite(f"{output_dir}/frame_{frame_idx:04d}.png", image)
            frame_idx += 1
        count += 1
    
    vidcap.release()
