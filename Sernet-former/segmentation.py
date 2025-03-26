import os
import torch
import torch.nn.functional as F
import torchvision.models.segmentation as segmentation
from torchvision.io.image import read_image
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
from torchvision.transforms.functional import to_pil_image
from PIL import Image
import numpy as np
import cv2
from scipy.ndimage import binary_dilation

def segmentation(image_path, save_dir="propainter_inputs/object_removal/mask"):
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    im1 = read_image(image_path).to(device)

    # Load DeepLabV3 model with pretrained weights
    weights = DeepLabV3_ResNet101_Weights.DEFAULT
    model = deeplabv3_resnet101(weights=weights).to(device)
    model.eval()
    
    # Apply preprocessing
    preprocess = weights.transforms()
    batch = preprocess(im1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        prediction = model(batch)["out"]

    # **Upsample the output to match the original image size**
    original_size = im1.shape[1:]  # (height, width)
    upsampled_mask = F.interpolate(prediction, size=original_size, mode="bilinear", align_corners=False)

    # Normalize and extract masks
    normalized_masks = upsampled_mask.softmax(dim=1).cpu()
    class_to_idx = {cls: idx for idx, cls in enumerate(weights.meta["categories"])}

    # Define the classes to extract
    target_classes = ("car", "bicycle", "bus", "motorbike", "person")

    # Initialize an empty mask
    combined_mask = torch.zeros_like(normalized_masks[0, 0])

    # Extract and combine masks for each class
    for class_name in target_classes:
        if class_name in class_to_idx:
            class_mask = normalized_masks[0, class_to_idx[class_name]]
            combined_mask = torch.logical_or(combined_mask, class_mask > 0.1)

    # Convert mask to NumPy
    mask_np = combined_mask.cpu().numpy().astype(np.bool_)

    # **Apply Binary Dilation to Expand the Mask**
    structure = np.ones((10, 10), dtype=np.bool_)  # Adjust size for more expansion
    mask_np_dilated = binary_dilation(mask_np, structure=structure)

    # Convert back to tensor
    combined_mask = torch.tensor(mask_np_dilated, dtype=torch.bool)

    # Get mask indices
    mask_indices = np.argwhere(mask_np_dilated)  # Extract (row, col) coordinates

    indices_dict = {
        "rows": mask_indices[:, 0],  # NumPy array of row indices
        "cols": mask_indices[:, 1],  # NumPy array of column indices
    }
    
    # # Convert to PIL image and save
    # mask_img = to_pil_image(combined_mask.float())
    # mask_filename = os.path.join(save_dir, f"{os.path.basename(image_path).split('.')[0]}_mask.png")
    # mask_img.save(mask_filename)
    # print(f"Mask saved at {mask_filename}")

    # # **Overlay Mask on Original Image**
    # original_image = Image.open(image_path).convert("RGB")
    # mask_np = np.array(mask_img)  # Convert mask to NumPy
    # mask_colored = np.zeros_like(original_image, dtype=np.uint8)  # Create a blank image

    # # Apply red color to the mask
    # mask_colored[mask_np > 0] = [255, 0, 0]  # Red mask
    
    # # Convert images to OpenCV format
    # original_cv = np.array(original_image)
    # overlayed_image = cv2.addWeighted(original_cv, 0.7, mask_colored, 0.3, 0)

    # # Save the overlay image
    # overlay_filename = os.path.join(save_dir, f"{os.path.basename(image_path).split('.')[0]}_overlay.png")
    # Image.fromarray(overlayed_image).save(overlay_filename)
    # print(f"Overlay image saved at {overlay_filename}")

    return indices_dict
