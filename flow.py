import numpy as np
import cv2
import argparse
import os

# Function to visualize optical flow
def viz_flow(flow, save_path=None):
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    if save_path:
        cv2.imwrite(save_path, bgr)
    return bgr

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow_dir", type=str, required=True, help="Path to directory containing .npy optical flow files")
    parser.add_argument("--output_dir", type=str, default=None, help="Path to save visualized images")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True) if args.output_dir else None

    for file in sorted(os.listdir(args.flow_dir)):
        if file.endswith(".npy"):
            flow_path = os.path.join(args.flow_dir, file)
            flow = np.load(flow_path)
            vis = viz_flow(flow)
            
            cv2.imshow("Optical Flow", vis)
            if args.output_dir:
                save_path = os.path.join(args.output_dir, file.replace(".npy", ".png"))
                cv2.imwrite(save_path, vis)
            
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
    
    cv2.destroyAllWindows()

