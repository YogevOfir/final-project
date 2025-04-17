#!/usr/bin/env python3
"""
Rotated Template Matching for Drone Image on a Big Map

Assumptions:
  - The big image (e.g., a semantic segmentation map) is 4K.
  - The small drone image is known to be a crop from the big image with dimensions:
      height = 620 pixels, width = 1100 pixels.
  - The drone image might be rotated with respect to the big image.
  - This script iterates over a set of rotation angles, performing template matching
    for each rotated template, and then selects the best match.

Usage:
    python rotatedTemplateMatching.py --big_map "path/to/big_map.png" --template "path/to/drone_image.png"
"""

import cv2
import numpy as np
import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser(
        description="Rotated Template Matching for Drone Image on a Big Map")
    parser.add_argument('--big_map', required=True, help='Path to the big image (4K).')
    parser.add_argument('--template', required=True, help='Path to the drone image (crop, 620x1100).')
    parser.add_argument('--angle_range', type=float, default=20.0,
                        help='Maximum rotation angle (in degrees) to try in each direction. Default is 20.')
    parser.add_argument('--angle_step', type=float, default=2.0,
                        help='Step (in degrees) for rotation search. Default is 2.')
    parser.add_argument('--method', default='TM_SQDIFF_NORMED',
                        help='Template matching method. Options: TM_CCOEFF, TM_CCOEFF_NORMED, TM_CCORR, TM_CCORR_NORMED, TM_SQDIFF, TM_SQDIFF_NORMED.')
    return parser.parse_args()

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
        print(f"Method {method_name} not recognized. Falling back to TM_SQDIFF_NORMED.")
        return cv2.TM_SQDIFF_NORMED
    return methods[method_name]

def rotate_image(image, angle):
    """Rotate image around its center."""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    # Use border mode to avoid black corners (could use cv2.BORDER_REPLICATE, etc.)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return rotated

def main():
    args = parse_args()

    # Load images
    big_img = cv2.imread(args.big_map, cv2.IMREAD_COLOR)
    template = cv2.imread(args.template, cv2.IMREAD_COLOR)

    if big_img is None:
        print(f"Error: Could not load big map image from {args.big_map}")
        sys.exit(1)
    if template is None:
        print(f"Error: Could not load template image from {args.template}")
        sys.exit(1)

    # Debug: print image sizes
    print("Big image shape:", big_img.shape)
    print("Drone image shape:", template.shape)

    # Optionally verify the template size is as expected:
    expected_h, expected_w = 620, 1100
    tem_h, tem_w = template.shape[:2]
    if (tem_h, tem_w) != (expected_h, expected_w):
        print(f"Warning: Expected template size ({expected_h}x{expected_w}), but got ({tem_h}x{tem_w}).")
    
    # Convert big image to grayscale (and template as well) for matching.
    big_gray = cv2.cvtColor(big_img, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # Prepare matching method
    method = get_matching_method(args.method)
    
    # Define angle search range (from -angle_range to +angle_range)
    angle_range = args.angle_range
    angle_step = args.angle_step
    angles = np.arange(-angle_range, angle_range+angle_step, angle_step)

    best_score = None
    best_angle = None
    best_loc = None
    best_result = None
    best_template = None

    for angle in angles:
        # Rotate the template by the current angle.
        rotated_template = rotate_image(template_gray, angle)
        # Note: if the rotated image remains the same size, parts of the image might be cut off.
        # If needed, one can compute a bounding rectangle that fully contains the rotated image.
        res = cv2.matchTemplate(big_gray, rotated_template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # For TM_SQDIFF_NORMED, a lower value is better.
        score = min_val if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED] else max_val

        if best_score is None or (method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED] and score < best_score) \
           or (method not in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED] and score > best_score):
            best_score = score
            best_angle = angle
            best_loc = min_loc if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED] else max_loc
            best_result = res.copy()
            best_template = rotated_template.copy()

    print(f"Best match found at angle {best_angle:.2f} degrees with score {best_score:.4f}.")
    # Determine bottom-right coordinate using the dimensions of the (rotated) template.
    tem_h, tem_w = best_template.shape
    bottom_right = (best_loc[0] + tem_w, best_loc[1] + tem_h)

    # Draw the rectangle on a copy of the big image.
    result_img = big_img.copy()
    cv2.rectangle(result_img, best_loc, bottom_right, (0, 255, 0), 3)

    # Display the matching result.
    cv2.namedWindow("Big Map with Detected Region", cv2.WINDOW_NORMAL)
    cv2.imshow("Big Map with Detected Region", result_img)

    cv2.namedWindow("Best Rotated Template", cv2.WINDOW_NORMAL)
    cv2.imshow("Best Rotated Template", best_template)

    cv2.namedWindow("Matching Score Map", cv2.WINDOW_NORMAL)
    cv2.imshow("Matching Score Map", best_result / best_result.max())  # normalize for visualization

    print("Press any key in one of the image windows to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
