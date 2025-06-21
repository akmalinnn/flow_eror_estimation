import os
import sys
import torch
import cv2
import numpy as np
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "core")))

from raft import RAFT
from raft_utils import flow_viz
from raft_utils.utils import InputPadder

DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def viz(flo, save_path=None, npy_path=None, ):
    # Ensure the flow tensor is in the expected dense format
    flo = flo[0].permute(1, 2, 0).cpu().numpy()  # Convert flow to (H, W, 2)


    if npy_path:
        np.save(npy_path, flo)
        print(f"Flow data saved at {npy_path}")

    # Map flow to an RGB image
    flo_rgb = flow_viz.flow_to_image(flo)

    # If a save path is provided, save the flow image
    if save_path:
        # Convert to BGR for OpenCV compatibility (if needed)
        flo_bgr = flo_rgb[:, :, [2, 1, 0]]  # OpenCV uses BGR by default
        cv2.imwrite(save_path, flo_bgr)  # Save as an image (e.g., PNG, JPEG)
        print(f"Flow image saved at {save_path}")

def demo(raft_opts, imfile1, imfile2, output_dir="output_flow"):
    # Load the RAFT model
    model = torch.nn.DataParallel(RAFT(raft_opts))
    model.load_state_dict(torch.load(raft_opts.model))
    model = model.module
    model.to(DEVICE)
    model.eval()

    os.makedirs(output_dir, exist_ok=True)  

    with torch.no_grad():
        # Load images
        image1 = load_image(imfile1)
        image2 = load_image(imfile2)

        # Pad images
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        # Compute optical flow
        flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)


        # Generate filename for saving the flow image
        flow_filename = os.path.join(output_dir, f"flow_{os.path.basename(imfile1)}")
        npy_filename = os.path.join(output_dir, f"flow_{os.path.basename(imfile1)}.npy")
        
        # Visualize and save the flow image
        # viz(flow_up, save_path=flow_filename)

        viz(flow_up, save_path=flow_filename, npy_path=npy_filename)
