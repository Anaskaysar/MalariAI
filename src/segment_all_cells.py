import cv2
import numpy as np
import argparse
import os

def segment_cells(image_path, output_path=None):
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found.")
        return
        
    img = cv2.imread(image_path)
    if img is None:
        print("Failed to load image.")
        return
        
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # The image has a bright background and dark cells.
    # Otsu's thresholding after inverse binary threshold
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Morphological operations to remove small noise and close holes 
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Dilation to combine nearby fragments
    sure_bg = cv2.dilate(opening, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(sure_bg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    output_img = img.copy()
    cell_count = 0
    boxes = []
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Filter out tiny noise and gigantic artifacts
        if 15 < w < 200 and 15 < h < 200:
            cv2.rectangle(output_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            boxes.append((x, y, w, h))
            cell_count += 1
            
    print(f"Detected {cell_count} cells using OpenCV segmentation.")
    
    if output_path:
        cv2.imwrite(output_path, output_img)
        print(f"Saved visualization to {output_path}")
        
    return boxes

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="ImageMal.jpg")
    parser.add_argument("--out", type=str, default="results/segmented_all_cv2.jpg")
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    segment_cells(args.image, args.out)
