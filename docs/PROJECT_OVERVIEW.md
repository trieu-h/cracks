# Crack Segmentation Model Overview

## Business Understanding

### Problem Statement

Extreme environments, continuous loads, and stress expose critical physical structures, machinery, and foundations to continuous surface degradation and structural cracking. Identifying and repairing these micro/macro fractures relies heavily on manual human inspection procedures, which can be hazardously slow and prone to subjective oversight, especially in hard-to-reach locations.

This leads to a dramatically higher chance of unexpected catastrophic failures, extensive repair downtime, and shortened equipment lifespans across the board.

### Goal

Creating an autonomous structural treatment and monitoring ecosystem utilizing modern AI segmentation models (mapped through a robust laboratory UI) to identify, track, and interact with crack propagations in real-time.

## Data Acquisition & Understanding & Preparation

### Public Data

Public datasets (e.g., from Roboflow or Kaggle) tracking concrete/metal crack segmentations are localized physically.

Dataset architecture necessitates utilizing the Ultralytics layout. The UI features a pipeline strictly looking for properly scaled `data.yaml` files bridging `train/val/test` arrays naturally, without requiring external script converters.

Directory: `./datasets`

### Proprietary Data

Proprietary field images integrate specific real-world stress scenarios. Proprietary drone or internal structural footage is actively unpacked frame-by-frame. 

Data processing handles:
- Utilizing segmentation interfaces assigning precise polygon boundaries over complex branching fractures rather than simple bounding boxes.
- Normalizing directories strictly towards the `yolo26`/`yolo11n-seg` input parameters. 

## Model Training

Pre-loaded variants revolving around **YOLOv26/YOLO11 Segmentation** architectures were strictly integrated as they drastically optimize real-time performance on edge endpoints (such as a Raspberry Pi 5 or embedded NVidia Jetsons). 

The initial pipeline trains specifically on one unified class: `crack`. Additional material variations can be added seamlessly via the dataset parser.

Example Training Bounds:
- Epochs: Adjustable up to 200 via the Training Interface.
- Batch Size: 16 (Dynamic scaling supported depending on GPU limits).
- Image Size: 640 standard mapped.
- Optimizer: Built-in PyTorch optimizations mapped (Adam/SGD tracking).

To instantiate processes or manipulate variables, invoke actions directly via the unified UI ecosystem (detailed rigorously in [DOCUMENTATION.md](../DOCUMENTATION.md)).

## UI

The project bridges a complex Full-stack UI (FastAPI + React 18) featuring an immersive 1970s laboratory environment to systematically manage the recurrent tasks organically: Checkpoint discovery, Live Web-Socket Training Telemetry, Prediction Inspections, and Dataset linking.

Please consult the primary [DOCUMENTATION.md](../DOCUMENTATION.md) for extended system maps.

## Versioning

Active checkpoints maintain two-point persistence: 
1. SQLite Database map caching metrics and historical logs natively.
2. Directly exported `.pt` matrices saved sequentially within `/checkpoints` and `runs/segment/train` local hubs.

Model versioning integrates live offline-sync hooks to automatically discover weights synthesized outside of the UI terminal organically.

## Future Improvements

- Deploy integrated pipeline directly onto a Raspberry Pi 5 standard kit.
- Optimization scaling specific for headless real-time analysis against drone/robotic feeds.
- Broaden target classes: Spalling, Delamination, Micro-Fissures.
- Cloud DB integration expanding beyond localized SQLite restrictions.

## FAQ

Frequented asked question documentation can be found in [FAQ.md](FAQ.md).
