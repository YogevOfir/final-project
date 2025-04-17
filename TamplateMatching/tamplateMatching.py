#!/usr/bin/env python3
"""
Template Matching Script for Semantic Segmented Maps with Scale Adjustment

This script loads a large semantic segmentation map and a drone (template) image,
rescales the template image based on the provided physical heights of the map and the drone image,
performs template matching, and displays:
  - The full map with a rectangle drawn around the matched region.
  - The cropped matched region.

Usage:
    python tamplateMatching.py --big_map "path/to/big_map.png" --template "path/to/template.png" [--method METHOD]
    
Assumptions (adjust these values if needed):
  - Big map covers a physical height of 250 meters.
  - Drone image covers a physical height of 100 meters.
"""

import cv2
import numpy as np
import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Template Matching for Semantic Segmented Maps with Scale Adjustment")
    parser.add_argument('--big_map',default=r"C:\Code&Proj\Final Proj\TamplateMatching\data\map\Map data\Label\label_for_fram_990.png", help='Path to the big semantic segmentation map image.')
    parser.add_argument('--template', default=r"C:\Code&Proj\Final Proj\TamplateMatching\data\map\check map\frame_990_1565_880.png", help='Path to the drone (template) image.')
    parser.add_argument('--method', default='TM_SQDIFF_NORMED',
                        help='Template matching method. Options: TM_CCOEFF, TM_CCOEFF_NORMED, TM_CCORR, TM_CCORR_NORMED, TM_SQDIFF, TM_SQDIFF_NORMED.')
    args = parser.parse_args()
    return args

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
        return cv2.TM_CCOEFF_NORMED
    return methods[method_name]

def main():
    args = parse_args()

    # Load images (both assumed to be 4K in resolution)
    big_img = cv2.imread(args.big_map, cv2.IMREAD_COLOR)
    template = cv2.imread(args.template, cv2.IMREAD_COLOR)

    if big_img is None:
        print(f"Error: Could not load big map image from {args.big_map}")
        sys.exit(1)
    if template is None:
        print(f"Error: Could not load template image from {args.template}")
        sys.exit(1)

    # Debug: print original image sizes
    print("Original big image shape:", big_img.shape)
    print("Original template image shape:", template.shape)

    # --- Scale the Template Image ---
    # Physical heights (in meters) for the images (change these values if needed)
    # map_physical_height = 1      # e.g., big map covers 250 meters
    # drone_physical_height = 1    # drone image covers 100 meters

    # # Calculate pixels-per-meter (ppm) for each image based on their heights.
    # map_ppm = big_img.shape[0] / map_physical_height
    # drone_ppm = template.shape[0] / drone_physical_height

    # # Calculate the scaling factor for the drone image to match the map's resolution.
    # scale_factor = map_ppm / drone_ppm
    # print(f"Map ppm: {map_ppm:.2f}, Drone ppm: {drone_ppm:.2f}")
    # print(f"Scaling factor for drone image: {scale_factor:.2f}")

    # if scale_factor != 1.0:
    #     new_width = int(template.shape[1] * scale_factor)
    #     new_height = int(template.shape[0] * scale_factor)
    #     template = cv2.resize(template, (new_width, new_height), interpolation=cv2.INTER_AREA)
    #     print("Scaled template image shape:", template.shape)

    # Convert images to grayscale for matching
    big_img_gray = cv2.cvtColor(big_img, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # Choose the matching method.
    method = get_matching_method(args.method)

    # Perform template matching.
    result = cv2.matchTemplate(big_img_gray, template_gray, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    # Depending on the method, decide the best match.
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
        match_value = min_val
    else:
        top_left = max_loc
        match_value = max_val

    # Determine bottom-right coordinate using the dimensions of the (now scaled) template.
    template_h, template_w = template_gray.shape
    bottom_right = (top_left[0] + template_w, top_left[1] + template_h)

    print("Match top-left:", top_left)
    print("Match bottom-right:", bottom_right)
    print("Matching score:", match_value)

    # Draw a rectangle on a copy of the big image to show the matched region.
    result_img = big_img.copy()
    cv2.rectangle(result_img, top_left, bottom_right, (0, 255, 0), 2)

    # Crop the matched region from the big image (ensure within boundaries).
    height, width, _ = big_img.shape
    x1 = max(top_left[0], 0)
    y1 = max(top_left[1], 0)
    x2 = min(bottom_right[0], width)
    y2 = min(bottom_right[1], height)
    matched_region = big_img[y1:y2, x1:x2]

    # --- Display the Results in Resizable Windows ---
    cv2.namedWindow("Big Map with Match", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Big Map with Match", 800, 600)
    cv2.imshow("Big Map with Match", result_img)

    cv2.namedWindow("Matched Region", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Matched Region", 400, 300)
    cv2.imshow("Matched Region", matched_region)

    print("Press any key in one of the image windows to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
