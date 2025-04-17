import numpy as np
import cv2
import os

# Define the RGB to Class mapping
def rgb_to_class(mask_rgb):
    rgb_mapping = {
        (0, 0, 0): 0,
        (102, 102, 156): 5,
        (128, 64, 128): 2,
        (107, 142, 35): 3,
        (0, 0, 142): 4,
        (70, 70, 70): 5,
        (0, 200, 200): 6,
    }
    h, w, _ = mask_rgb.shape
    class_mask = np.zeros((h, w), dtype=np.uint8)
    
    for rgb, cls in rgb_mapping.items():
        class_mask[np.all(mask_rgb == rgb, axis=-1)] = cls
    
    return class_mask

# Initialize accumulators for IoU and F1 calculations
num_classes = 6
intersection_accumulator = np.zeros(num_classes, dtype=np.uint64)
union_accumulator = np.zeros(num_classes, dtype=np.uint64)
true_positive = np.zeros(num_classes, dtype=np.uint64)
false_positive = np.zeros(num_classes, dtype=np.uint64)
false_negative = np.zeros(num_classes, dtype=np.uint64)

# Paths to Ground Truth and Predicted Labels
ground_truth_root = r"C:\Code&Proj\Final Proj\data\Experiments\div_by_res\360p"
predicted_root = r"fig_results\uavid\tabel1_1024_down&up_scaling\360p"

# Process all sequences and images
for seq in os.listdir(ground_truth_root):
    print("comparing seq: ", seq)
    gt_seq_path = os.path.join(ground_truth_root, seq, "Labels")
    pred_seq_path = os.path.join(predicted_root, seq, "Labels")

    if not os.path.isdir(gt_seq_path) or not os.path.isdir(pred_seq_path):
        print(f"Skipping {seq} because directories don't exist.")
        continue

    # Iterate through ground truth files
    for gt_file in os.listdir(gt_seq_path):
        print("compare file: ", gt_file)
        if gt_file.endswith(".png"):  # Only process PNG files
            gt_path = os.path.join(gt_seq_path, gt_file)
            pred_path = os.path.join(pred_seq_path, gt_file)

            # Check if predicted file exists
            if not os.path.isfile(pred_path):
                print(f"Warning: Predicted file not found for {gt_file}")
                continue

            # Load ground truth and predicted images
            gt_rgb = cv2.imread(gt_path)
            pred_rgb = cv2.imread(pred_path)

            if gt_rgb is None or pred_rgb is None:
                print(f"Error loading images for {gt_file}")
                continue

            # Convert to RGB
            gt_rgb = cv2.cvtColor(gt_rgb, cv2.COLOR_BGR2RGB)
            pred_rgb = cv2.cvtColor(pred_rgb, cv2.COLOR_BGR2RGB)

            # Convert RGB masks to class masks
            gt_mask = rgb_to_class(gt_rgb)
            pred_mask = rgb_to_class(pred_rgb)

            # Update accumulators for IoU and F1 Score
            for cls in range(num_classes):
                pred = (pred_mask == cls)
                gt = (gt_mask == cls)

                intersection = np.logical_and(pred, gt).sum()
                union = np.logical_or(pred, gt).sum()

                # Accumulate values
                intersection_accumulator[cls] += intersection
                union_accumulator[cls] += union

                true_positive[cls] += intersection
                false_positive[cls] += (pred.sum() - intersection)
                false_negative[cls] += (gt.sum() - intersection)

# Calculate overall IoU and F1 Score for each class
iou_per_class = []
f1_per_class = []

for cls in range(num_classes):
    intersection = intersection_accumulator[cls]
    union = union_accumulator[cls]

    # IoU Calculation
    iou = intersection / union if union > 0 else np.nan
    iou_per_class.append(iou)

    # F1 Score Calculation
    precision = true_positive[cls] / (true_positive[cls] + false_positive[cls]) if (true_positive[cls] + false_positive[cls]) > 0 else 0
    recall = true_positive[cls] / (true_positive[cls] + false_negative[cls]) if (true_positive[cls] + false_negative[cls]) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else np.nan
    f1_per_class.append(f1)

# Calculate mean IoU and mean F1 Score
mean_iou = np.nanmean(iou_per_class)
mean_f1 = np.nanmean(f1_per_class)

# Print Results
print("\nOverall IoU per class:")
for i, iou in enumerate(iou_per_class):
    print(f"  Class {i}: {iou:.4f}" if not np.isnan(iou) else f"  Class {i}: Not present")

print("\nOverall F1 Score per class:")
for i, f1 in enumerate(f1_per_class):
    print(f"  Class {i}: {f1:.4f}" if not np.isnan(f1) else f"  Class {i}: Not present")

print(f"\nMean IoU: {mean_iou:.4f}")
print(f"Mean F1 Score: {mean_f1:.4f}")
