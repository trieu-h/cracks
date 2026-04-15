# Crack Detection Laboratory - Complete Documentation

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [User Guide](#user-guide)
4. [Troubleshooting](#troubleshooting)

---

## Overview

This is a full-stack web application designed to help users train AI models to accurately detect cracks and faults in structural materials using modern YOLO architectures. The system is wrapped in a vintage 1970s laboratory visual aesthetic.

### Key Features
- **Unified Pipeline**: Manage datasets, train models, and test inferences securely from one centralized application.
- **Persistent History**: The backend saves all session checkpoints to a local SQLite database seamlessly preventing progress loss.
- **Live Metrics**: Monitor live charts to understand how the model's loss improves iteration by iteration.
- **Versatile Testing**: Supports uploading single images or full `.mp4` video files to let the AI highlight faults.

## Architecture

The system utilizes an decoupled architecture separating frontend interfaces and backend processing:

- **Backend Network**: Python 3.10 and FastAPI control HTTP endpoints and WebSocket streams. PyTorch relies on the physical GPU to process tensor math.
- **Storage**: A SQLite database manages relationships between imported datasets and active training cycles.
- **Frontend Dashboard**: React 18, utilizing TailwindCSS, renders the views and requests API updates dynamically.
- **AI Models**: Integrates Ultralytics frameworks specifically tailored for the YOLO Segmentation standard layout.

---

## User Guide

### 1. Adding Datasets
To prepare the system, tell it where your images are located.
- Navigate to the **Datasets** tab.
- Submit the absolute path of your local images folder.
- Execute **Import**. The backend reads your files and securely maps them into the database for selection later.

### 2. Training Control
- Open the **Training** tab.
- Select your previously imported dataset.
- Click **Start Training**. The server will take over the math calculation.
- The interface will display real-time charts logging progress seamlessly as the AI optimizes.

### 3. Running Detection Tests
- Has the model completed its learning phase successfully? Let's verify it.
- Open the **Detection** tab.
- Upload an image or video file.
- Click **Run Detection**. The AI will process the file natively and overlay precision boundaries over any structural fractures it identifies.

---

## Troubleshooting

### Missing Files During Dataset Import
Verify that you typed the directory location accurately. Using incorrect slashes between Windows and Mac structures frequently causes parsing failures.

### Out of Memory (OOM) Errors or Frozen Charts
Sometimes your GPU consumes too much memory. Ensure you do not have intensive background applications active. If the backend fails abruptly, simply restart the application process. Because checkpoints are stored locally, you can load the **Training** tab and click **Resume** to automatically pick up where it broke.

### Mac Silicon Capability
The codebase functions correctly on Apple Silicon (M-series) systems. However, they lack NVIDIA GPU components. Therefore, the background model training relies purely on system CPUs, resulting in much slower performance metrics compared to a standard NVIDIA-equipped environment.
