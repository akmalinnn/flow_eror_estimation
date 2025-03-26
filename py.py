import numpy as np
import argparse
import os

def load_and_display_npy(npy_file):
    if not os.path.exists(npy_file):
        print(f"File not found: {npy_file}")
        return
    
    data = np.load(npy_file)
    print("Shape:", data.shape)
    print("Top-left 5x5 section:")
    print(data[:5, :5])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and display an npy file.")
    parser.add_argument("npy_file", type=str, help="Path to the .npy file")
    args = parser.parse_args()
    
    load_and_display_npy(args.npy_file)
