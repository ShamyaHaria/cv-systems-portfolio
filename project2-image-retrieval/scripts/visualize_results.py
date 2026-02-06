#!/usr/bin/env python3
"""
Author: Shamya Haria
Date: February 5, 2026
Purpose: Visualizes retrieval results as a grid of target and matched images
"""

import cv2
import numpy as np
import sys
import os

def create_result_grid(target_path, match_paths, output_path, title="Results"):
    """Creates 2x3 grid showing target image and top 5 matches"""
    target = cv2.imread(target_path)
    if target is None:
        print(f"Error: Could not read {target_path}")
        return
    
    # Resize to standard dimensions
    target_height = 300
    aspect_ratio = target.shape[1] / target.shape[0]
    target_width = int(target_height * aspect_ratio)
    target_resized = cv2.resize(target, (target_width, target_height))
    cv2.putText(target_resized, "Target", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Process match images
    match_images = []
    for i, match_path in enumerate(match_paths[:5]):
        img = cv2.imread(match_path)
        if img is not None:
            img_resized = cv2.resize(img, (target_width, target_height))
            cv2.putText(img_resized, f"Match {i+1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            match_images.append(img_resized)
    
    # Arrange in 2x3 grid
    row1 = [target_resized] + match_images[:2]
    row2 = match_images[2:5]
    
    while len(row2) < 3:
        row2.append(np.zeros_like(target_resized))
    
    top_row = np.hstack(row1)
    bottom_row = np.hstack(row2)
    grid = np.vstack([top_row, bottom_row])
    
    # Add title bar
    title_height = 50
    title_img = np.ones((title_height, grid.shape[1], 3), dtype=np.uint8) * 255
    cv2.putText(title_img, title, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    
    final = np.vstack([title_img, grid])
    cv2.imwrite(output_path, final)
    print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python3 visualize_results.py <target> <match1> <match2> ... <output>")
        sys.exit(1)
    
    target = sys.argv[1]
    matches = sys.argv[2:-1]
    output = sys.argv[-1]
    
    create_result_grid(target, matches, output)