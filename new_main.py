import sys
import os
import torch
import subprocess
from argparse import Namespace
import numpy as np
import cv2

sys.path.append(os.path.abspath("Raft"))
sys.path.append(os.path.abspath("Sernet-former"))

from rdemo import demo as run_raft_demo
from segmentation import segmentation as run_sernet_segmentation
from raft_utils import flow_viz 

# Paths
INPUT_FOLDER = "input_videos"
OUTPUT_FRAMES_FOLDER = "extracted_frames"
FLOW_IMAGE_FOLDER = "flow"
FLOW_DIFF_FOLDER = "flow_diff"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Ensure necessary folders exist
os.makedirs(FLOW_IMAGE_FOLDER, exist_ok=True)
os.makedirs(FLOW_DIFF_FOLDER, exist_ok=True)

def extract_frames(video_path, output_folder, width=640, height=360):
    os.makedirs(output_folder, exist_ok=True)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    ffmpeg_cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", f"scale={width}:{height}",
        os.path.join(output_folder, f"{video_name}_frame_%05d.png")
    ]
    subprocess.run(ffmpeg_cmd, check=True)
    print(f"Extracted frames for {video_name} in {output_folder}")

def process_video(video):
    video_path = os.path.join(INPUT_FOLDER, video)
    video_name = os.path.splitext(video)[0]
    
    video_frame_folder = os.path.join(OUTPUT_FRAMES_FOLDER, video_name)
    video_flow_folder = os.path.join(FLOW_IMAGE_FOLDER, video_name)
    video_diff_folder = os.path.join(FLOW_DIFF_FOLDER, video_name)
    
    os.makedirs(video_frame_folder, exist_ok=True)
    os.makedirs(video_flow_folder, exist_ok=True)
    os.makedirs(video_diff_folder, exist_ok=True)
    
    extract_frames(video_path, video_frame_folder)
    
    raft_opts = Namespace(
        small=False,
        mixed_precision=False,
        alternate_corr=False,
        model="Raft/models/raft-kitti.pth",
        path=video_frame_folder,
    )
    
    frame_files = sorted(os.listdir(video_frame_folder))
    for i in range(len(frame_files) - 1):
        frame1_path = os.path.join(video_frame_folder, frame_files[i])
        frame2_path = os.path.join(video_frame_folder, frame_files[i + 1])
        run_raft_demo(raft_opts, frame1_path, frame2_path, video_flow_folder)
    
    save_frames(video_frame_folder, video_flow_folder, video_name)
    subtract_npy_files(video_flow_folder, video_diff_folder, video_frame_folder)
    print(f"Processing completed for {video_name}")

def save_frames(input_folder, output_folder, video_name):
    os.makedirs(output_folder, exist_ok=True)
    images = sorted([img for img in os.listdir(input_folder) if img.endswith(".png")])
    npy_files = sorted([npy for npy in os.listdir(input_folder) if npy.endswith(".npy")])
    
    if not images or not npy_files:
        print(f"No images or npy files found in {input_folder}")
        return
    
    for i, npy_name in enumerate(npy_files):
        npy_path = os.path.join(input_folder, npy_name)
        npy_output_path = os.path.join(output_folder, f"{video_name}_frame_{i+1:05d}.npy")
        np.save(npy_output_path, np.load(npy_path))
    
    print(f"Saved frames and npy duplicates in: {output_folder}")

def subtract_npy_files(input_folder, output_folder, frame_folder):
    os.makedirs(output_folder, exist_ok=True)
    npy_files = sorted([f for f in os.listdir(input_folder) if f.endswith(".npy")])
    frame_files = sorted(os.listdir(frame_folder))
    
    if len(npy_files) < 2 or len(frame_files) < 2:
        print("Not enough npy files or frames to perform subtraction.")
        return
    
    for i in range(len(npy_files) - 1):
        npy1_path = os.path.join(input_folder, npy_files[i])
        npy2_path = os.path.join(input_folder, npy_files[i + 1])
        frame2_path = os.path.join(frame_folder, frame_files[i + 1])
        
        indices_dict = run_sernet_segmentation(frame2_path)  
        data1 = np.load(npy1_path)
        data2 = np.load(npy2_path)
        
        if "rows" in indices_dict and "cols" in indices_dict:
            rows = indices_dict["rows"]
            cols = indices_dict["cols"]
            data1[rows, cols, :] = 0
            data2[rows, cols, :] = 0
        
        difference = data1 - data2
        output_filename = f"diff_{npy_files[i].replace('.npy', '')}.npy"
        output_path = os.path.join(output_folder, output_filename)
        np.save(output_path, difference)
    
        difference_img = flow_viz.flow_to_image(difference)
        img_output_path = os.path.join(output_folder, f"{output_filename}.png")
        cv2.imwrite(img_output_path, difference_img)
    
        print(f"Saved difference: {output_path}")

def main():
    video_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".mp4")]
    if not video_files:
        print("No videos found in input folder.")
        return
    
    for video in video_files:
        process_video(video)

if __name__ == "__main__":
    main()
