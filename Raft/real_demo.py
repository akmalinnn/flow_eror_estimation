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

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def viz(flo, save_path=None, npy_path=None):
    """ Visualizes and saves the optical flow. """
    flo = flo[0].permute(1, 2, 0).cpu().numpy()  # Convert to (H, W, 2)

    if npy_path:
        np.save(npy_path, flo)
        print(f"Flow data saved at {npy_path}")

    # Convert flow to an RGB image
    flo_rgb = flow_viz.flow_to_image(flo)

    # Save flow image if required
    if save_path:
        flo_bgr = flo_rgb[:, :, [2, 1, 0]]  # Convert RGB → BGR for OpenCV
        cv2.imwrite(save_path, flo_bgr)
        print(f"Flow image saved at {save_path}")

def demo(raft_opts, imfile1, imfile2, indices_dict, output_dir="output_flow"):
    """ Runs RAFT and removes optical flow at mask locations. """

    # Load the RAFT model
    model = torch.nn.DataParallel(RAFT(raft_opts))
    model.load_state_dict(torch.load(raft_opts.model, map_location=DEVICE))
    model = model.module.to(DEVICE).eval()

    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        # Load images
        image1 = load_image(imfile1)
        image2 = load_image(imfile2)

        # Pad images
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        # Compute optical flow
        _, flow_up = model(image1, image2, iters=20, test_mode=True)

        # Clone flow tensor to apply filtering
        flow_up_filtered = flow_up.clone()

        # Ensure indices exist and remove flow at mask locations
        if "rows" in indices_dict and "cols" in indices_dict:
            rows, cols = indices_dict["rows"], indices_dict["cols"]
            flow_up_filtered[:, :, rows, cols] = 0  # Zero out flow at mask locations

        # Generate filenames
        base_name1 = os.path.splitext(os.path.basename(imfile1))[0]
        base_name2 = os.path.splitext(os.path.basename(imfile2))[0]
        flow_filename = os.path.join(output_dir, f"flow_{base_name1}_{base_name2}.png")
        npy_filename = os.path.join(output_dir, f"flow_{base_name1}_{base_name2}.npy")

        # Visualize and save the modified flow image
        viz(flow_up_filtered, save_path=flow_filename, npy_path=npy_filename)

    print(f"Filtered optical flow saved at {flow_filename} and {npy_filename}")
