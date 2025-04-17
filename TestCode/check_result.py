import os
import cv2
import numpy as np

##############################################################################
# CONFIGURATION
##############################################################################

# Root directories for predictions and ground truth
PRED_ROOT = r"fig_results\uavid\unetformer_r18_png"
GT_ROOT   = r"data\uavid\uavid_test_old"

# Number of classes
NUM_CLASSES = 7

# Define your color map in RGB order
COLOR_MAP_RGB = {
    (0,   0,   0): 0,    # Background
    (102, 102, 156): 1,  # Class 1
    (128, 64, 128): 2,   # Class 2
    (107, 142, 35): 3,   # Class 3
    (0,   0, 142): 4,    # Class 4
    (70,  70,  70): 5,    # Class 5
    (0, 200, 200): 6,     # Class 6
}

# Specify the sequence you want to evaluate, e.g., 'seq10'
SEQUENCE = "seq10"

##############################################################################
# HELPER FUNCTIONS
##############################################################################

def color_to_label_id(img_rgb, color_map):
    """
    Convert a color-coded (H, W, 3) RGB image to a single-channel
    label mask (H, W), where each pixel is the class ID.
    """
    h, w, _ = img_rgb.shape
    label_mask = np.zeros((h, w), dtype=np.uint8)

    for rgb_color, class_id in color_map.items():
        # Create a boolean mask where the image matches this color exactly
        match = np.all(img_rgb == rgb_color, axis=-1)
        label_mask[match] = class_id

    return label_mask

def compute_ious_per_class(pred_mask, gt_mask, num_classes):
    """
    Computes IoU for each class from 0 to num_classes-1.
    Returns a list of length num_classes.
    """
    ious = []
    for c in range(num_classes):
        pred_c = (pred_mask == c)
        gt_c   = (gt_mask == c)
        intersection = np.logical_and(pred_c, gt_c).sum()
        union        = pred_c.sum() + gt_c.sum() - intersection

        if union == 0:
            # If both pred and gt have zero pixels for class c, define IoU=1.0
            iou_c = 1.0 if pred_c.sum() == 0 and gt_c.sum() == 0 else 0.0
        else:
            iou_c = intersection / union

        ious.append(iou_c)
    return ious

##############################################################################
# MAIN FUNCTION
##############################################################################

def compute_miou_sequence(pred_seq_dir, gt_seq_dir, color_map, num_classes):
    """
    Computes per-class IoU and mean IoU over all image pairs in a sequence.
    
    Parameters:
    - pred_seq_dir: Directory containing prediction label images.
    - gt_seq_dir: Directory containing ground truth label images.
    - color_map: Dictionary mapping RGB tuples to class IDs.
    - num_classes: Total number of classes.
    
    Returns:
    - mean_ious: List of mean IoU per class.
    - overall_miou: Overall mean IoU across all classes.
    """
    # List all image files in prediction directory
    pred_files = sorted([
        f for f in os.listdir(pred_seq_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
    ])

    if not pred_files:
        print(f"No prediction images found in {pred_seq_dir}.")
        return

    # Initialize lists to accumulate IoUs per class
    ious_per_class = [[] for _ in range(num_classes)]
    total_images = 0

    for filename in pred_files:
        pred_path = os.path.join(pred_seq_dir, filename)
        gt_path   = os.path.join(gt_seq_dir, filename)

        if not os.path.exists(gt_path):
            print(f"[Warning] Ground truth not found for {filename}. Skipping.")
            continue

        # Read images as BGR
        pred_bgr = cv2.imread(pred_path, cv2.IMREAD_COLOR)
        gt_bgr   = cv2.imread(gt_path,   cv2.IMREAD_COLOR)

        if pred_bgr is None or gt_bgr is None:
            print(f"[Warning] Could not read {filename}. Skipping.")
            continue

        # Convert BGR to RGB
        pred_rgb = cv2.cvtColor(pred_bgr, cv2.COLOR_BGR2RGB)
        gt_rgb   = cv2.cvtColor(gt_bgr,   cv2.COLOR_BGR2RGB)

        # Convert color-coded images to label IDs
        pred_mask = color_to_label_id(pred_rgb, color_map)
        gt_mask   = color_to_label_id(gt_rgb,   color_map)

        # Sanity check: shapes match
        if pred_mask.shape != gt_mask.shape:
            print(f"[Warning] Shape mismatch for {filename}. Skipping.")
            continue

        # Compute IoU for this image
        ious = compute_ious_per_class(pred_mask, gt_mask, num_classes)

        # Accumulate IoUs per class
        for c in range(num_classes):
            ious_per_class[c].append(ious[c])

        total_images += 1

    if total_images == 0:
        print("No valid image pairs processed.")
        return

    # Compute mean IoU per class
    mean_ious = []
    for c in range(num_classes):
        if ious_per_class[c]:
            mean_iou = np.mean(ious_per_class[c])
        else:
            mean_iou = float('nan')  # Undefined if no data
        mean_ious.append(mean_iou)

    # Compute overall mean IoU, ignoring NaNs
    overall_miou = np.nanmean(mean_ious)

    return mean_ious, overall_miou

##############################################################################
# EXECUTION
##############################################################################

if __name__ == "__main__":
    # Define color map (ensure it matches your label encoding)
    COLOR_MAP_RGB = {
        (0,   0,   0): 0,    # Background
        (156, 102, 102): 1,  # Class 1
        (128, 64, 128): 2,   # Class 2
        (35, 142,107): 3,   # Class 3
        (142, 0,   0 ): 4,    # Class 4
        (70,  70,  70): 5,    # Class 5
        ( 200, 200,0): 6,     # Class 6
    }

    # Specify the sequence
    SEQUENCE = "seq130"

    # Define prediction and ground truth directories for the sequence
    pred_seq_dir = os.path.join(PRED_ROOT, SEQUENCE, "Labels")
    gt_seq_dir   = os.path.join(GT_ROOT, SEQUENCE, "Labels")

    # Check if directories exist
    if not os.path.isdir(pred_seq_dir):
        print(f"Prediction directory does not exist: {pred_seq_dir}")
        exit(1)
    if not os.path.isdir(gt_seq_dir):
        print(f"Ground truth directory does not exist: {gt_seq_dir}")
        exit(1)

    # Compute mIoU
    mean_ious, overall_miou = compute_miou_sequence(
        pred_seq_dir,
        gt_seq_dir,
        COLOR_MAP_RGB,
        NUM_CLASSES
    )

    if mean_ious is not None:
        print("-----------------------------------")
        print(f"Sequence: {SEQUENCE}")
        print("IoU per class:")
        for class_id, iou in enumerate(mean_ious):
            if not np.isnan(iou):
                print(f"  Class {class_id}: IoU = {iou:.4f}")
            else:
                print(f"  Class {class_id}: IoU = Undefined (no samples)")
        print("-----------------------------------")
        print(f"Overall Mean IoU over {NUM_CLASSES} classes: {overall_miou:.4f}")
        print(f"Total images processed: {len(os.listdir(pred_seq_dir))}")
