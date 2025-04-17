# Semantic Template Matching for Map-Based UAV Localization

## Overview

This repository implements a method for **localizing a UAV** (Unmanned Aerial Vehicle) by combining **deep learning-based semantic segmentation** with **image template matching** on georeferenced maps. The pipeline works as follows:

1. **Semantic Segmentation**: A pretrained segmentation model (e.g., UnetFormer) identifies classes like roads, buildings, and vegetation from the UAV’s onboard camera image.
2. **Template Extraction**: The segmented output (semantic mask) serves as a template capturing the scene’s structural layout.
3. **Map Matching**: This semantic template is matched against a larger ground-truth map (with the same semantic labeling) to find the best location alignment.

By harnessing semantic cues instead of raw RGB data, the approach improves robustness to lighting, seasonal changes, and appearance variations.

## Project Structure

```
final-project/
├── GeoSeg/                    # Semantic segmentation module
│   ├── config/                # Dataset and training configs
│   ├── tools/                 # Preprocessing utilities (patch splitting, mask conversions)
│   ├── train_supervision.py   # Model training script
│   ├── inference_*.py         # Inference scripts for segmentation
│   └── requirements.txt       # Python dependencies for segmentation

├── TamplateMatching/          # Template matching module
│   ├── tamplateMatching.py                # Core matching pipeline
│   ├── tamplateMatching_with_labels.py    # Matching using semantic labels
│   ├── tamplateMatching_with_angle_BF.py  # Brute-force angle handling
│   ├── UAVlocation.py                     # High-level localization script
│   └── data/                  # Example map & UAV image data

├── TestCode/                  # Utility and test scripts
│   ├── change_test_to_png.py  # Data format conversion
│   └── check_result.py        # Visualization helpers

├── lightning_logs/            # PyTorch Lightning training logs & checkpoints
│   └── uavid/                 # Logs for UAVid dataset experiments

├── Semantic Template Matching For Map-Based UAV Localization.pdf  # Project report
├── presentation.pptx          # Project presentation slides
└── README.md                  # This file
```

## Installation and Setup

Follow these steps to prepare your environment and run the code:

1. **Clone the repository**
   ```bash
   git clone https://github.com/moshenh01/final-project.git
   cd final-project
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python3 -m venv venv
   source venv/bin/activate    # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r GeoSeg/requirements.txt
   ```
   > **Note**: If PyTorch is not included in `requirements.txt`, install it separately via:
   > ```bash
   > pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   > ```

4. **Verify the setup**
   ```bash
   python -c "import torch, pytorch_lightning, cv2; print('Setup OK')"
   ```

You’re now ready to train the segmentation model (`train_supervision.py`) or run localization experiments (`tamplateMatching.py` and variants). For detailed usage and parameter options, refer to the script docstrings or the project report PDF.