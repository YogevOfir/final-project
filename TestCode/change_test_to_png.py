import os
import cv2
import shutil

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
GT_ROOT = r"fig_results\uavid\unetformer_r18_old"   # where your .jpg labels currently reside
OUTPUT_ROOT = r"fig_results\uavid\unetformer_r18_png"  # where .png files will be saved
# If you have single-channel label masks to preserve exactly, use IMREAD_UNCHANGED.
# For color images, you might use IMREAD_COLOR. Adjust as needed.
LOAD_FLAG = cv2.IMREAD_UNCHANGED


def convert_and_copy_labels(gt_root, out_root):
    """
    Recursively walks `gt_root`.
    - If a file is .jpg/.jpeg, it converts to .png and saves in `out_root`.
    - If a file is .png, it copies it to `out_root`.
    Subfolder structure is preserved.
    """
    for root, dirs, files in os.walk(gt_root):
        # 1) Figure out the subfolder path relative to GT_ROOT
        #    e.g., if root = "GT_ROOT/seq1/Labels" => rel_path = "seq1/Labels"
        rel_path = os.path.relpath(root, gt_root)

        # 2) Create the corresponding subfolder in OUTPUT_ROOT
        out_subdir = os.path.join(out_root, rel_path)
        os.makedirs(out_subdir, exist_ok=True)

        # 3) Iterate over files
        for filename in files:
            fname_lower = filename.lower()
            full_old_path = os.path.join(root, filename)

            # 3a) If it's .jpg or .jpeg, load & convert to .png
            if fname_lower.endswith(".jpg") or fname_lower.endswith(".jpeg"):
                base_name = os.path.splitext(filename)[0]
                new_filename = base_name + ".png"
                full_new_path = os.path.join(out_subdir, new_filename)

                # Load image (single-channel or color, depending on LOAD_FLAG)
                img = cv2.imread(full_old_path, LOAD_FLAG)
                if img is None:
                    print(f"[Warning] Could not read: {full_old_path}. Skipped.")
                    continue

                success = cv2.imwrite(full_new_path, img)
                if success:
                    print(f"Converted: {full_old_path} -> {full_new_path}")
                else:
                    print(f"[Warning] Failed to save PNG for {full_old_path}")

            # 3b) If it's already .png, just copy it
            elif fname_lower.endswith(".png"):
                full_new_path = os.path.join(out_subdir, filename)
                shutil.copy2(full_old_path, full_new_path)
                print(f"Copied: {full_old_path} -> {full_new_path}")

            else:
                # If you want to handle other extensions (e.g. .tif) similarly,
                # you can add more conditions here.
                pass


def main():
    convert_and_copy_labels(GT_ROOT, OUTPUT_ROOT)


if __name__ == "__main__":
    main()
