#!/usr/bin/env python3
"""
Template Matching Script for Semantic Segmented Maps with Semantic Masking

This script loads a large semantic segmentation map and a drone (template) image.
It creates a combined semantic mask that:
  - Ignores the "trees_veg" class.
  - Merges the "building_roof" and "building_wall" classes.
Then, it applies the mask to both images and performs template matching.
Finally, it displays:
  - The full map with a rectangle drawn around the matched region.
  - The cropped matched region.

Usage:
    python tamplateMatching.py --big_map "path/to/big_map.png" --template "path/to/template.png" [--method METHOD]

Make sure to install OpenCV (e.g., via: pip install opencv-python)
"""

import os
import cv2
import numpy as np
import argparse
import sys
import time 

# -----------------------------------------------------------------------------
# Define semantic class colors (BGR format) – adjust these if needed.
# -----------------------------------------------------------------------------
CLASS_COLORS = {
    "building_roof": (70, 70, 70),
    "building_wall": (156,102,102),
    "road":          (128,64,128),
    "trees_veg":     (35,142,107),
    "background":    (0, 0, 0)
}

# -----------------------------------------------------------------------------
# Create binary masks and combine them.
# -----------------------------------------------------------------------------
def create_binary_masks(image, class_colors, tolerance=10):
    """
    Create a dictionary mapping each (or groups of) class(es) to a binary mask.
    
    In this function:
      - The "building_roof" and "building_wall" classes are merged into a single mask ("building").
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
    building_mask = None  # To hold the union of building_roof and building_wall.
    for cls, color in class_colors.items():
        if cls == "trees_veg":
            continue  # Ignore trees_veg.
        lower = np.array([max(c - tolerance, 0) for c in color], dtype=np.uint8)
        upper = np.array([min(c + tolerance, 255) for c in color], dtype=np.uint8)
        mask = cv2.inRange(image, lower, upper)
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
    The mask is the union of all masks created by create_binary_masks.
    
    Parameters:
      image, class_colors, tolerance: See create_binary_masks.
      
    Returns:
      combined: A binary mask (uint8) with 255 for pixels belonging to any desired class.
    """
    masks = create_binary_masks(image, class_colors, tolerance)
    combined = np.zeros(image.shape[:2], dtype=np.uint8)
    for key in masks:
        combined = cv2.bitwise_or(combined, masks[key])
    return combined

# -----------------------------------------------------------------------------
# Parse command-line arguments.
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Template Matching for Semantic Segmented Maps with Semantic Masking")
    parser.add_argument('--big_map', default=r"C:\Code&Proj\Final Proj\TamplateMatching\data\map\Map data\Label\label_for_007200.png",
                        help='Path to the big semantic segmentation map image.')
    parser.add_argument('--template', default=r"C:\Code&Proj\Final Proj\TamplateMatching\data\map\check map\007200_1136.png",
                        help='Path to the drone (template) image.')
    parser.add_argument('--method', default='TM_SQDIFF_NORMED',
                        help='Template matching method. Options: TM_CCOEFF, TM_CCOEFF_NORMED, TM_CCORR, TM_CCORR_NORMED, TM_SQDIFF, TM_SQDIFF_NORMED.')
    parser.add_argument('--tolerance', type=int, default=10, help="Color tolerance for semantic mask creation (default: 10).")
    args = parser.parse_args()
    return args

# -----------------------------------------------------------------------------
# Main routine.
# -----------------------------------------------------------------------------
def main():
    args = parse_args()

    # Load images.
    big_img = cv2.imread(args.big_map, cv2.IMREAD_COLOR)
    template = cv2.imread(args.template, cv2.IMREAD_COLOR)

    if big_img is None:
        print(f"Error: Could not load big map image from {args.big_map}")
        sys.exit(1)
    if template is None:
        print(f"Error: Could not load template image from {args.template}")
        sys.exit(1)

    # Debug: print original image sizes.
    print("Original big image shape:", big_img.shape)
    print("Original template image shape:", template.shape)

    # Convert images to grayscale.
    big_img_gray = cv2.cvtColor(big_img, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # Create semantic masks for both images.
    big_mask = create_combined_mask(big_img, CLASS_COLORS, tolerance=args.tolerance)
    template_mask = create_combined_mask(template, CLASS_COLORS, tolerance=args.tolerance)

    # Apply the masks to the grayscale images.
    masked_big = cv2.bitwise_and(big_img_gray, big_mask)
    masked_template = cv2.bitwise_and(template_gray, template_mask)

    # Choose the matching method.
    def get_matching_method(method_name):
        methods = {
            'TM_CCOEFF': cv2.TM_CCOEFF,
            'TM_CCOEFF_NORMED': cv2.TM_CCOEFF_NORMED,
            'TM_CCORR': cv2.TM_CCORR,
            'TM_CCORR_NORMED': cv2.TM_CCORR_NORMED,
            'TM_SQDIFF': cv2.TM_SQDIFF,
            'TM_SQDIFF_NORMED': cv2.TM_SQDIFF_NORMED
        }
        if method_name not in methods:
            print(f"Method {method_name} not recognized. Falling back to TM_CCOEFF_NORMED.")
            return cv2.TM_COEFF_NORMED
        return methods[method_name]

    method = get_matching_method(args.method)

    start_time = time.time()
    # Perform template matching on the masked images.
    result = cv2.matchTemplate(masked_big, masked_template, method)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Template matching took {elapsed_time:.4f} seconds.")
    
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # For TM_SQDIFF and TM_SQDIFF_NORMED, lower is better.
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
        match_value = min_val
    else:
        top_left = max_loc
        match_value = max_val

    # Determine bottom-right coordinate using the dimensions of the template.
    template_h, template_w = template_gray.shape
    bottom_right = (top_left[0] + template_w, top_left[1] + template_h)

    print("Match top-left:", top_left)
    print("Match bottom-right:", bottom_right)
    print("Matching score:", match_value)

    # Draw a rectangle on a copy of the big image to indicate the matched region.
    result_img = big_img.copy()
    cv2.rectangle(result_img, top_left, bottom_right, (0, 255, 0), 2)

    # Crop the matched region from the big image.
    height, width, _ = big_img.shape
    x1 = max(top_left[0], 0)
    y1 = max(top_left[1], 0)
    x2 = min(bottom_right[0], width)
    y2 = min(bottom_right[1], height)
    matched_region = big_img[y1:y2, x1:x2]

    # Display the results in resizable windows.
    cv2.namedWindow("Big Map with Match", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Big Map with Match", 1080, 1080)
    cv2.imshow("Big Map with Match", result_img)

    cv2.namedWindow("Matched Region", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Matched Region", 960, 540)
    cv2.imshow("Matched Region", matched_region)
    
    # # Save the images to the output folder.
    # output_folder = r"TamplateMatching\data\Experiment\frame_990"
    # os.makedirs(output_folder, exist_ok=True)
    # cv2.imwrite(os.path.join(output_folder, "big_map_with_match.png"), result_img)
    # cv2.imwrite(os.path.join(output_folder, "matched_region.png"), matched_region)


    print("Press any key in the image window to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
