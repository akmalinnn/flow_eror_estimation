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
OUTPUT_FLOW_FRAMES_FOLDER = "output_flow"
FLOW_IMAGE_FOLDER = "flow"
FLOW_DIFF_FOLDER = "flow_diff"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Ensure necessary folders exist
os.makedirs(FLOW_IMAGE_FOLDER, exist_ok=True)
os.makedirs(FLOW_DIFF_FOLDER, exist_ok=True)


# Frame Extraction
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


# Save Frames with Duplicates
def save_frames(input_folder, output_folder, video_name):
    os.makedirs(output_folder, exist_ok=True)
    
    # Get sorted image and npy file lists
    images = sorted([img for img in os.listdir(input_folder) if img.endswith(".png")])
    npy_files = sorted([npy for npy in os.listdir(input_folder) if npy.endswith(".npy")])

    if not images or not npy_files:
        print(f"No images or npy files found in {input_folder}")
        return

    frame_index = 1
    for i, (img_name, npy_name) in enumerate(zip(images, npy_files)):
        img_path = os.path.join(input_folder, img_name)
        npy_path = os.path.join(input_folder, npy_name)

        # # Save image
        # frame = cv2.imread(img_path)
        # frame_output_path = os.path.join(output_folder, f"{video_name}_frame_{frame_index:05d}.jpg")
        # cv2.imwrite(frame_output_path, frame)

        # Save npy file
        npy_data = np.load(npy_path)
        npy_output_path = os.path.join(output_folder, f"{video_name}_frame_{frame_index:05d}.npy")
        np.save(npy_output_path, npy_data)

        frame_index += 1

        

    print(f"Saved frames and npy duplicates in: {output_folder}")


# Clean Folder
def clean_folder(folder_path):
    if os.path.exists(folder_path):
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                os.rmdir(file_path)
        print(f"Cleaned folder: {folder_path}")


import matplotlib.pyplot as plt

def visualize_flow(flow, title="Optical Flow (RAFT Style)"):
    """Visualize optical flow using RAFT's flow_viz module."""
    flow_img = flow_viz.flow_to_image(flow)  # Convert flow to RGB image
    
    plt.figure(figsize=(10, 5))
    plt.imshow(flow_img)
    plt.axis("off")
    plt.title(title)
    plt.show()

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
        frame2_path = os.path.join(frame_folder, frame_files[i + 1])  # Use frame2 for segmentation
        
        indices_dict = run_sernet_segmentation(frame2_path)  
        # Load optical flow data
        data1 = np.load(npy1_path)  # Shape: (H, W, 2)
        data2 = np.load(npy2_path)

        print(f"Processing: {npy_files[i]} & {npy_files[i+1]} | Shape: {data1.shape}")

        if "rows" in indices_dict and "cols" in indices_dict:
            rows = indices_dict["rows"]
            cols = indices_dict["cols"]
            data1[rows, cols, :] = 0  # Zero out masked pixels in data1
            data2[rows, cols, :] = 0  # Zero out masked pixels in data2

        # Visualize original optical flows
        # visualize_flow(data1, title="Optical Flow - Frame 1")
        # visualize_flow(data2, title="Optical Flow - Frame 2")

        # Compute optical flow difference
        difference = data1 - data2

        # Save the difference
        output_filename = f"diff_{npy_files[i].replace('.npy', '')}_{npy_files[i+1].replace('.npy', '')}.npy"
        output_path = os.path.join(output_folder, output_filename)
        np.save(output_path, difference)

        # Convert to image and save as PNG
        difference_img = flow_viz.flow_to_image(difference)
        img_output_path = os.path.join(output_folder, f"{output_filename}.png")
        cv2.imwrite(img_output_path, difference_img)

        # Visualize the difference
        # visualize_flow(difference, title="Optical Flow Difference (Masked)")

        print(f"Saved difference: {output_path}")



# Main Batch Processing
def main():
    video_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".mp4")]
    if not video_files:
        print("No videos found in input folder.")
        return

    for video in video_files:
        video_path = os.path.join(INPUT_FOLDER, video)
        video_name = os.path.splitext(video)[0]
        
        extract_frames(video_path, OUTPUT_FRAMES_FOLDER)
        
        os.makedirs(OUTPUT_FLOW_FRAMES_FOLDER, exist_ok=True)

        raft_opts = Namespace(
            small=False,
            mixed_precision=False,
            alternate_corr=False,
            model="Raft/models/raft-kitti.pth",
            path=OUTPUT_FRAMES_FOLDER,
        )

        frame_files = sorted(os.listdir(OUTPUT_FRAMES_FOLDER))
        for i in range(len(frame_files) - 1):
            frame1_path = os.path.join(OUTPUT_FRAMES_FOLDER, frame_files[i])
            frame2_path = os.path.join(OUTPUT_FRAMES_FOLDER, frame_files[i + 1])

            # Run segmentation before passing to RAFT
            # indices_dict = run_sernet_segmentation(frame2_path)

            # Run RAFT with zeroed flow values
            run_raft_demo(raft_opts, frame1_path, frame2_path)
        
        save_frames(OUTPUT_FLOW_FRAMES_FOLDER, os.path.join(FLOW_IMAGE_FOLDER, video_name), video_name)
        # Run NPY subtraction with segmentation masking

        clean_folder(OUTPUT_FLOW_FRAMES_FOLDER)

    for subfolder in os.listdir(FLOW_IMAGE_FOLDER):
        subfolder_path = os.path.join(FLOW_IMAGE_FOLDER, subfolder)
        if os.path.isdir(subfolder_path):
            print(f"Processing subtraction for folder: {subfolder_path}")
            subtract_npy_files(subfolder_path, FLOW_DIFF_FOLDER, OUTPUT_FRAMES_FOLDER)

    clean_folder(OUTPUT_FRAMES_FOLDER)


if __name__ == "__main__":
    main()
