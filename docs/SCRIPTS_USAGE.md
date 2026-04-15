# Script Explanations

This document offers brief, mid-level explanations for the primary background server services powering the application UI. You do not need developer-level expertise to understand the overall software hooks.

## 1. Backend Training Pipeline (`training.py`)

This file manages the model's learning execution cycles.

**Main functions:**
- `start_training_session`: Triggers the training sequence. It parses your GUI variables (epochs, batch size), establishes a unique session ID, and safely pushes execution off the main application thread to prevent UI freezing.
- `stop_training_session`: This function securely interrupts an active run. Upon receiving a stop signal, it halts the model gracefully ensuring the generated weight components `.pt` are not corrupted.

---

## 2. Video Parsing Pipeline (`video_detection.py`)

This file extracts and constructs video evaluations natively.

**Main functions:**
- `extract_frames`: Slices an uploaded `.mp4` into individual `.jpg` frames utilizing OpenCV.
- `detect_video_frames`: Cycles through each extracted image recursively running standard YOLO evaluation predicting the coordinates of the structural faults.
- `create_annotated_video`: Once the frames contain overlaid defect markings, the script bundles the frames back together into a final compressed `.mp4` video format suitable for playback directly inside the frontend component.

---

## 3. The Offline Synchronization Hook (`sync.py`)

This file is responsible for finding lost or externally trained records.

**Main functions:**
- `discover_offline_runs`: If you share or transfer completed runs between colleagues externally, you can simply paste the directory structures directly into the project `/checkpoints` hub. This script scans the folder contents, parses standard `results.csv` outputs, and integrates the external history logs natively into your app interface without needing to manually map it.
