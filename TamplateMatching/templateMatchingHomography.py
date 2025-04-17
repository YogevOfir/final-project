#!/usr/bin/env python3
"""
Semantic Segmentation Matching with Masked Sliding Window Homography
---------------------------------------------------------------------

This script attempts to locate a drone (template) segmentation image within a large
segmentation map using sliding window homography matching. Before matching, both the big map 
and the template are masked based on semantic segmentation information:
  - The "building_roof" and "building_wall" classes are merged into a single "building" mask.
  - The "trees_veg" class is ignored.
  
ORB features are computed on the masked grayscale images, and for each candidate window
(with the same dimensions as the template) in the big image, we estimate a homography via RANSAC.
The window with the highest number of inlier matches is selected as the best match.

Usage:
    python semantic_matching.py --big_map "path/to/big_map.png" --template "path/to/template.png" 
       --method homography_masked [--stride 20] [--min_matches 4] [--ransac_thresh 5.0] [--tolerance 10]

Install OpenCV with:
    pip install opencv-python
"""

import cv2
import numpy as np
import argparse

# -----------------------------------------------------------------------------
# Define semantic class colors (BGR format)
# -----------------------------------------------------------------------------
CLASS_COLORS = {
    "building_roof": (70, 70, 70),
    "building_wall": (156, 102, 102),
    "road":          (128, 64, 128),
    "trees_veg":     (35, 142, 107),
    "background":    (0, 0, 0)
}

# -----------------------------------------------------------------------------
# Functions for creating semantic masks
# -----------------------------------------------------------------------------
def create_binary_masks(image, class_colors, tolerance=10):
    """
    Create a dictionary mapping each (or groups of) class(es) to a binary mask.
    In this function:
      - The "building_roof" and "building_wall" classes are merged into a single "building" mask.
      - The "trees_veg" class is ignored.
      - All other classes are processed normally.
      
    Parameters:
      image       : Input color image.
      class_colors: Dictionary of class_name -> (B, G, R) tuple.
      tolerance   : Allowed deviation per channel.
      
    Returns:
      masks: Dictionary of combined classes -> binary mask (uint8, values 0 or 255).
    """
    masks = {}
    building_mask = None  # Will hold the union of building_roof and building_wall.
    
    for cls, color in class_colors.items():
        # Skip the "trees_veg" class.
        if cls == "trees_veg":
            continue

        lower = np.array([max(c - tolerance, 0) for c in color], dtype=np.uint8)
        upper = np.array([min(c + tolerance, 255) for c in color], dtype=np.uint8)
        mask = cv2.inRange(image, lower, upper)  # Binary mask: 0 or 255
        
        if cls in ["building_roof", "building_wall"]:
            if building_mask is None:
                building_mask = mask
            else:
                building_mask = cv2.bitwise_or(building_mask, mask)
        else:
            masks[cls] = mask
    if building_mask is not None:
        masks["building"] = building_mask
    return masks

def create_combined_mask(image, class_colors, tolerance=10):
    """
    Create a single combined binary mask from the semantic segmentation.
    This mask is the union of all masks created by create_binary_masks.
    
    Parameters:
      image, class_colors, tolerance: As in create_binary_masks.
      
    Returns:
      combined: A binary mask (uint8) with 255 for pixels belonging to any desired class.
    """
    masks = create_binary_masks(image, class_colors, tolerance)
    combined = np.zeros(image.shape[:2], dtype=np.uint8)
    for key, m in masks.items():
        combined = cv2.bitwise_or(combined, m)
    return combined

# -----------------------------------------------------------------------------
# Sliding Window Homography Matching (Masked)
# -----------------------------------------------------------------------------
def sliding_window_homography_masked(big_img, template_img, stride=20, min_matches=4, ransac_thresh=5.0, tolerance=10):
    """
    Slide a window (of the same size as the template) over the big image and, for each window,
    compute a homography between the masked template and the masked candidate window using ORB feature matching.
    The candidate window with the highest number of inlier matches is selected as the best match.
    
    Both the big image and the template image are pre-masked using semantic segmentation information.
    
    Parameters:
      big_img: Big segmentation map (BGR).
      template_img: Drone segmentation image (BGR).
      stride: Sliding window stride (in pixels).
      min_matches: Minimum number of ORB matches required.
      ransac_thresh: RANSAC reprojection threshold.
      tolerance: Color tolerance for mask creation.
      
    Returns:
      best_loc: (x, y) top-left coordinates of the best matching window.
      best_H: Homography matrix (template-to-window).
      best_inliers: Number of inlier matches (score).
    """
    # Create combined semantic masks for both images.
    big_mask = create_combined_mask(big_img, CLASS_COLORS, tolerance)
    template_mask = create_combined_mask(template_img, CLASS_COLORS, tolerance)
    
    # Convert images to grayscale.
    big_gray = cv2.cvtColor(big_img, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
    
    # Apply the masks to focus on desired regions.
    masked_big = cv2.bitwise_and(big_gray, big_mask)
    masked_template = cv2.bitwise_and(template_gray, template_mask)
    
    # Initialize ORB detector and BFMatcher.
    orb = cv2.ORB_create(nfeatures=10000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # Compute features for the masked template.
    kp_template, des_template = orb.detectAndCompute(masked_template, None)
    if des_template is None or len(kp_template) < min_matches:
        print("Not enough features in the masked template.")
        return None, None, 0
    
    t_h, t_w = template_img.shape[:2]
    big_h, big_w = big_img.shape[:2]
    
    best_inliers = 0
    best_H = None
    best_loc = None
    
    # Slide the window over the big image.
    for y in range(0, big_h - t_h + 1, stride):
        for x in range(0, big_w - t_w + 1, stride):
            window = big_img[y:y+t_h, x:x+t_w]
            window_gray = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)
            window_mask = create_combined_mask(window, CLASS_COLORS, tolerance)
            masked_window = cv2.bitwise_and(window_gray, window_mask)
            
            kp_window, des_window = orb.detectAndCompute(masked_window, None)
            if des_window is None or len(kp_window) < min_matches:
                continue
            
            matches = bf.match(des_template, des_window)
            if len(matches) < min_matches:
                continue
            
            matches = sorted(matches, key=lambda m: m.distance)
            pts_template = np.float32([kp_template[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            pts_window = np.float32([kp_window[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            
            H, mask = cv2.findHomography(pts_template, pts_window, cv2.RANSAC, ransac_thresh)
            if H is None or mask is None:
                continue
            inliers = int(np.sum(mask))
            if inliers > best_inliers:
                best_inliers = inliers
                best_H = H
                best_loc = (x, y)
                
    return best_loc, best_H, best_inliers

# -----------------------------------------------------------------------------
# Main Routine
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Semantic Segmentation Matching with Masked Sliding Window Homography")
    parser.add_argument('--big_map', required=True, help="Path to the big segmentation map image (e.g., 4K).")
    parser.add_argument('--template', required=True, help="Path to the drone template segmentation image (e.g., 620x1100).")
    parser.add_argument('--method', choices=['homography_masked'], default='homography_masked',
                        help="Matching method: 'homography_masked' for sliding window homography using semantic masks.")
    parser.add_argument('--stride', type=int, default=100, help="Sliding window stride (default: 20).")
    parser.add_argument('--min_matches', type=int, default=4, help="Minimum ORB matches required (default: 4).")
    parser.add_argument('--ransac_thresh', type=float, default=5.0, help="RANSAC reprojection threshold (default: 5.0).")
    parser.add_argument('--tolerance', type=int, default=10, help="Color tolerance for semantic mask creation (default: 10).")
    args = parser.parse_args()
    
    # Load images.
    big_img = cv2.imread(args.big_map)
    template_img = cv2.imread(args.template)
    if big_img is None or template_img is None:
        print("Error: Could not load one or both images.")
        return
    
    print("Big image shape:", big_img.shape)
    print("Template image shape:", template_img.shape)
    
    if args.method == 'homography_masked':
        print("Using masked sliding window homography matching (no rotation, angle = 0).")
        best_loc, best_H, best_inliers = sliding_window_homography_masked(big_img, template_img,
                                                                          stride=args.stride,
                                                                          min_matches=args.min_matches,
                                                                          ransac_thresh=args.ransac_thresh,
                                                                          tolerance=args.tolerance)
        if best_loc is None:
            print("No good match found.")
            return
        print(f"Best match found at location {best_loc} with {best_inliers} inliers.")
        t_h, t_w = template_img.shape[:2]
        result_img = big_img.copy()
        cv2.rectangle(result_img, best_loc, (best_loc[0] + t_w, best_loc[1] + t_h), (0, 255, 0), 3)
        cv2.namedWindow("Result (homography_masked)", cv2.WINDOW_NORMAL)
        cv2.imshow("Result (homography_masked)", result_img)
    
    print("Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
