# Shamya Haria, CS 5330
# Project 5 Extension 3 - live webcam digit recognition
# Points camera at handwritten digits and classifies them in real time.
# Press Q to quit, S for screenshot.
# April 5 2026

import sys
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import cv2
import os

from task1_train import MyNetwork


def preprocess_roi(roi):
    """Crop a region of interest from the frame, convert to 28x28 grayscale,
    and normalize for the MNIST model."""
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi

    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)

    # if background is light, invert (MNIST expects white-on-black)
    if resized.mean() > 128:
        resized = 255 - resized

    arr = resized.astype(np.float32) / 255.0
    arr = (arr - 0.1307) / 0.3081
    tensor = torch.tensor(arr).unsqueeze(0).unsqueeze(0)
    return tensor


def find_digit_contours(gray_frame):
    """Use edge detection + contours to find regions that might contain digits."""
    blurred = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    digit_regions = []
    h_frame, w_frame = gray_frame.shape
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # filter by size and aspect ratio - digits are roughly tall rectangles
        if (40 < w < w_frame // 2 and 50 < h < h_frame // 2
                and 0.3 < h / w < 4.0):
            digit_regions.append((x, y, w, h))

    return digit_regions


def main():
    print('--- Extension 3: Live Digit Recognition ---')
    print('  Q/ESC = quit, S = screenshot, H = hold/resume')
    print()

    model_path = 'outputs/mnist_model.pth'
    if not os.path.exists(model_path):
        print('ERROR: model not found. Run task1_train.py first.')
        return

    model = MyNetwork()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    print('Model loaded, opening camera...')

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print('Could not open webcam. Need a camera connected.')
        return

    os.makedirs('outputs', exist_ok=True)
    hold = False
    held_frame = None
    screenshot_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print('Camera read failed')
            break

        if not hold:
            display_frame = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            digit_regions = find_digit_contours(gray)

            for (x, y, w, h) in digit_regions:
                pad = 5
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(frame.shape[1], x + w + pad)
                y2 = min(frame.shape[0], y + h + pad)
                roi = gray[y1:y2, x1:x2]

                try:
                    tensor = preprocess_roi(roi)
                    with torch.no_grad():
                        output = model(tensor)
                        probs = torch.exp(output)
                        confidence, pred = probs.max(dim=1)

                    # draw bounding box and label
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label_text = f'{pred.item()} ({confidence.item():.0%})'
                    cv2.putText(display_frame, label_text, (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                except:
                    pass  # skip if ROI is too small or something goes wrong

            # show status
            status = 'LIVE' if not hold else 'HELD'
            cv2.putText(display_frame, f'[{status}] {len(digit_regions)} regions',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            held_frame = display_frame
        else:
            display_frame = held_frame if held_frame is not None else frame

        cv2.imshow('Live Digit Recognition', display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key in [ord('q'), 27]:
            break
        elif key == ord('s'):
            fname = f'outputs/live_screenshot_{screenshot_num:03d}.png'
            cv2.imwrite(fname, display_frame)
            print(f'Saved screenshot: {fname}')
            screenshot_num += 1
        elif key == ord('h'):
            hold = not hold
            print('Detection', 'paused' if hold else 'resumed')

    cap.release()
    cv2.destroyAllWindows()
    print('Done')


if __name__ == '__main__':
    main()
