"""
Shamya Haria
February 23, 2026
CNN embedding classification using ResNet18 for improved accuracy
"""

import cv2
import numpy as np
import os
from glob import glob

def custom_threshold(image, threshold=100):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    binary = np.where(gray < threshold, 0, 255).astype(np.uint8)
    return binary

def morphological_cleanup(binary):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

def segment_regions(binary):
    return cv2.connectedComponentsWithStats(binary, connectivity=8)

def preprocess_for_cnn(original, labels, stats, centroids, region_id):
    cx, cy = centroids[region_id]
    x, y, w, h = stats[region_id][:4]
    
    region_mask = (labels == region_id).astype(np.uint8)
    moments = cv2.moments(region_mask)
    angle = 0.5 * np.arctan2(2 * moments['mu11'], moments['mu20'] - moments['mu02'])
    
    center = (int(cx), int(cy))
    rotation_matrix = cv2.getRotationMatrix2D(center, np.degrees(angle), 1.0)
    rotated = cv2.warpAffine(original, rotation_matrix, (original.shape[1], original.shape[0]))
    
    padding = 20
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(rotated.shape[1], x + w + padding)
    y2 = min(rotated.shape[0], y + h + padding)
    
    roi = rotated[y1:y2, x1:x2]
    return cv2.resize(roi, (224, 224))

def compute_embedding(image, net):
    blob = cv2.dnn.blobFromImage(image, 1.0/255.0, (224, 224), 
                                  (0.485, 0.456, 0.406), swapRB=True, crop=False)
    net.setInput(blob)
    return net.forward(net.getUnconnectedOutLayersNames()[0]).flatten()

def extract_label(filename):
    fn = os.path.basename(filename).lower()
    labels = {
        'triangle': 'triangle', 'hammer': 'hammer', 'allen': 'allen_key',
        'screwdriver': 'screwdriver', 'key_fob': 'key_fob', 'keyfob': 'key_fob',
        'pen': 'pen', 'phone': 'phone', 'mouse': 'mouse', 
        'glove': 'glove', 'bracelet': 'bracelet', 'scissors': 'scissors',
        'cd': 'cd', 'star': 'star', 'postit': 'postit', 'post-it': 'postit', 'box': 'box'
    }
    for key, value in labels.items():
        if key in fn:
            return value
    return 'unknown'

def process_image(path, net, threshold=100):
    img = cv2.imread(path)
    if img is None:
        return None
    
    binary = custom_threshold(img, threshold)
    cleaned = morphological_cleanup(binary)
    num_regions, labels, stats, centroids = segment_regions(cleaned)
    
    largest = -1
    max_area = 0
    for i in range(1, num_regions):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > max_area and area > 300:
            max_area = area
            largest = i
    
    if largest == -1:
        return None
    
    preprocessed = preprocess_for_cnn(img, labels, stats, centroids, largest)
    return compute_embedding(preprocessed, net)

print("=== CNN Embedding Evaluation ===\n")

try:
    net = cv2.dnn.readNetFromONNX("resnet18.onnx")
    print("✓ ResNet18 loaded\n")
except:
    print("❌ ResNet18 model not found!")
    print("Download: curl -L https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet18-v2-7.onnx -o resnet18.onnx")
    exit(1)

# Build training database
train_db = {}
for folder in ["data/train/"]:
    if not os.path.exists(folder):
        continue
    for img_path in glob(os.path.join(folder, "*.*")):
        if not img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        
        label = extract_label(img_path)
        if label == 'unknown':
            continue
        
        emb = process_image(img_path, net)
        if emb is not None:
            train_db.setdefault(label, []).append(emb)
            print(f"  Trained: {os.path.basename(img_path)} → {label}")

print(f"\n✓ Training complete: {len(train_db)} categories\n")

# Test
if os.path.exists("data/test/"):
    correct = 0
    total = 0
    
    for img_path in sorted(glob("data/test/*.*")):
        if not img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        
        true_label = extract_label(img_path)
        if true_label == 'unknown':
            continue
        
        emb = process_image(img_path, net)
        if emb is None:
            continue
        
        min_dist = float('inf')
        pred_label = 'unknown'
        
        for label, train_embs in train_db.items():
            for train_emb in train_embs:
                dist = np.linalg.norm(emb - train_emb)
                if dist < min_dist:
                    min_dist = dist
                    pred_label = label
        
        total += 1
        if pred_label == true_label:
            correct += 1
            print(f"✓ {os.path.basename(img_path)}: {true_label}")
        else:
            print(f"✗ {os.path.basename(img_path)}: {true_label} → {pred_label}")
    
    accuracy = (correct / total * 100) if total > 0 else 0
    print(f"\n=== Results ===")
    print(f"Accuracy: {accuracy:.1f}% ({correct}/{total})")
else:
    print("No test folder found - skipping evaluation")
EOF
</parameter>
<parameter name="description">Create CNN embedding evaluation script compatible with new system</parameter>