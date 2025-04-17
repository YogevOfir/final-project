#!/usr/bin/env python3
"""
Feature Matching for Drone Semantic Segmentation on a Big Map (Ratio-Based)
---------------------------------------------------------------------------
This script loads a large semantic segmentation map (big map) and a drone (template) image.
It rescales the drone image so that its height becomes 620 pixels (and its width scales accordingly,
approximately 1100 pixels if the aspect ratio is preserved). Then it uses ORB feature detection and
descriptor matching (with BFMatcher) to find corresponding features between the drone image and the big map.
A homography is estimated via RANSAC so that the drone image can be projected onto the big map,
thus handling rotation, scaling, and orientation differences.

Usage:
    python featureMatching.py --big_map "path/to/big_map.png" --template "path/to/drone_image.png"
    
Assumptions:
  - Big map is 4K (2160 x 3840).
  - The desired drone (template) crop is 620 pixels in height and ~1100 pixels in width.
"""

import cv2
import numpy as np
import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser(
        description="Feature Matching for Drone Segmentation on a Big Map using Homography (Ratio-Based)")
    parser.add_argument('--big_map', default=r"C:\Code&Proj\Final Proj\TamplateMatching\data\map\Map data\Label\label_for_7200.png",
                        help='Path to the big semantic segmentation map image.')
    parser.add_argument('--template',  default=r"C:\Code&Proj\Final Proj\TamplateMatching\data\map\check map\007200_720_1660_940.png",
                        help='Path to the drone (template) image.')
    args = parser.parse_args()
    return args

def scale_template(template):
    """
    Rescale the drone (template) image based solely on the ratio.
    We know that the drone image should correspond to a crop of size 620x1100 pixels.
    Here we compute a scale factor to force the template height to 620 pixels.
    (Assuming that the drone image is originally 4K like the big map.)
    """
    TARGET_HEIGHT = 620
    # Compute scale factor based on height.
    scale_factor = TARGET_HEIGHT / template.shape[0]
    new_width = int(template.shape[1] * scale_factor)
    new_height = int(template.shape[0] * scale_factor)
    print(f"Scaling drone image with factor: {scale_factor:.3f}")
    print(f"Scaled drone image shape: ({new_height}, {new_width})")
    return cv2.resize(template, (new_width, new_height), interpolation=cv2.INTER_AREA)

def feature_matching(big_img, template):
    """
    Uses ORB features and brute-force matching to find correspondences between the template
    and the big map. Estimates a homography using RANSAC.
    """
    # Convert images to grayscale (ORB works on grayscale)
    gray_big = cv2.cvtColor(big_img, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    # Initialize ORB detector (increase max features if needed)
    orb = cv2.ORB_create(nfeatures=10000)
    
    # Detect keypoints and compute descriptors
    kp_template, des_template = orb.detectAndCompute(gray_template, None)
    kp_big, des_big = orb.detectAndCompute(gray_big, None)
    
    if des_template is None or des_big is None:
        print("Not enough features detected in one or both images. Exiting.")
        sys.exit(1)
    
    # Create BFMatcher with Hamming distance (suitable for ORB)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # Match descriptors
    matches = bf.match(des_template, des_big)
    print(f"Found {len(matches)} raw matches.")
    
    # Sort matches by distance (lower distance is better)
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Select a subset of good matches (e.g., top 10% or at least 4)
    num_good = max(4, int(len(matches) * 0.1))
    good_matches = matches[:num_good]
    print(f"Using {len(good_matches)} good matches for homography estimation.")
    
    # Draw matches for visualization (optional)
    img_matches = cv2.drawMatches(template, kp_template, big_img, kp_big, good_matches, None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.namedWindow("Feature Matches", cv2.WINDOW_NORMAL)
    cv2.imshow("Feature Matches", img_matches)
    
    # Extract matched keypoints
    pts_template = np.float32([kp_template[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
    pts_big = np.float32([kp_big[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
    
    # Estimate homography using RANSAC
    H, mask = cv2.findHomography(pts_template, pts_big, cv2.RANSAC, 5.0)
    if H is None:
        print("Homography could not be computed. Exiting.")
        sys.exit(1)
    
    # Draw the projected template boundary on the big image
    h, w = template.shape[:2]
    pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, H)
    
    big_img_with_homography = big_img.copy()
    cv2.polylines(big_img_with_homography, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
    
    cv2.namedWindow("Detected Template", cv2.WINDOW_NORMAL)
    cv2.imshow("Detected Template", big_img_with_homography)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    args = parse_args()
    
    # Load images
    big_img = cv2.imread(args.big_map, cv2.IMREAD_COLOR)
    template = cv2.imread(args.template, cv2.IMREAD_COLOR)
    
    if big_img is None:
        print("Error loading big map image.")
        sys.exit(1)
    if template is None:
        print("Error loading drone image.")
        sys.exit(1)
    
    print("Big map shape:", big_img.shape)
    print("Drone image shape:", template.shape)
    
    # Scale the drone image based solely on the desired ratio
    #template_scaled = scale_template(template)
    
    # Perform feature matching and homography estimation
    feature_matching(big_img, template)
    
if __name__ == '__main__':
    main()
