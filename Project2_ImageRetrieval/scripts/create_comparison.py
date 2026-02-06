#!/usr/bin/env python3
"""
Author: Shamya Haria
Date: February 5, 2026
Purpose: Creates comparison grid showing different methods on same query image
"""

import cv2
import numpy as np

def create_comparison_grid(target_path, method_results, output_path):
    """Creates grid comparing multiple methods side by side"""
    target = cv2.imread(target_path)
    if target is None:
        print(f"Error: Could not read {target_path}")
        return
    
    size = 200
    target_resized = cv2.resize(target, (size, size))
    cv2.putText(target_resized, "Target", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    rows = [target_resized]
    
    # Create row for each method
    for method_name, match_paths in method_results.items():
        method_row = []
        for i, match_path in enumerate(match_paths[:3]):
            img = cv2.imread(match_path)
            if img is not None:
                img_resized = cv2.resize(img, (size, size))
                label = f"{method_name} #{i+1}"
                cv2.putText(img_resized, label, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                method_row.append(img_resized)
        
        if method_row:
            row_img = np.hstack(method_row)
            label_img = np.ones((size, 150, 3), dtype=np.uint8) * 255
            cv2.putText(label_img, method_name, (10, size//2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            row_with_label = np.hstack([label_img, row_img])
            rows.append(row_with_label)
    
    # Pad rows to same width and stack vertically
    max_width = max(r.shape[1] for r in rows)
    padded_rows = []
    for row in rows:
        if row.shape[1] < max_width:
            padding = np.ones((row.shape[0], max_width - row.shape[1], 3), dtype=np.uint8) * 255
            row = np.hstack([row, padding])
        padded_rows.append(row)
    
    final = np.vstack(padded_rows)
    cv2.imwrite(output_path, final)
    print(f"Saved comparison to {output_path}")

if __name__ == "__main__":
    target = "data/olympus/pic.1072.jpg"
    
    results = {
        "Baseline": ["data/olympus/pic.0768.jpg", "data/olympus/pic.0138.jpg", "data/olympus/pic.0234.jpg"],
        "Histogram": ["data/olympus/pic.0937.jpg", "data/olympus/pic.0142.jpg", "data/olympus/pic.0813.jpg"],
        "Multi-Hist": ["data/olympus/pic.0813.jpg", "data/olympus/pic.0701.jpg", "data/olympus/pic.0937.jpg"],
        "Texture+Color": ["data/olympus/pic.0701.jpg", "data/olympus/pic.0234.jpg", "data/olympus/pic.0430.jpg"],
        "DNN": ["data/olympus/pic.0143.jpg", "data/olympus/pic.0863.jpg", "data/olympus/pic.0234.jpg"]
    }
    
    create_comparison_grid(target, results, "results/task6/comparison_pic1072.jpg")