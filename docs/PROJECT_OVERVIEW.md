# Crack Detection Project Overview

## Problem Statement

Large structures, machinery, and physical foundations constantly suffer from surface degradation and structural cracking due to extreme environments and continuous stress. Identifying and repairing these fractures relies heavily on manual human inspection procedures. These practices are often slow, hazardous, and prone to subjective oversight, particularly in hard-to-reach locations.

If cracks go unnoticed, there is a significantly higher chance of unexpected system failures, extensive repair downtimes, and shortened hardware lifespans.

## Our Solution

We developed a comprehensive web application featuring a vintage industrial laboratory interface. This software utilizes standard AI segmentation models to identify, track, and highlight fault propagations across images and video feeds in real-time.

## Dataset Preparation

Training a reliable model requires standardized datasets of cracked surfaces.
- We support the standard **Ultralytics YOLO** folder architecture.
- Users can import public datasets or integrate proprietary images captured from drone infrastructure inspections.
- By utilizing advanced segmentation (polygon outlines) rather than simple bounding boxes, the system accurately maps the precise branching of complex fractures.

## Model Training

The pipeline strictly revolves around the **YOLO Segmentation** family (e.g., YOLOv11/YOLOv26). These models are highly optimized to run quickly on edge devices, such as a Raspberry Pi 5 or embedded NVidia Jetsons, without sacrificing deep accuracy.

The standard procedure involves selecting hyperparameters (Epochs, Batch Size) via the frontend, which the FastAPI backend translates into a PyTorch execution sequence. The progress is relayed visually back to the user via WebSockets in real-time.

## Application Interface

The project offers a Full-Stack UI built with React 18 and FastAPI. Its core features include:
- **Datasets**: Simple ingestion workflows for scanning folders.
- **Training**: Centralized control for executing and pausing PyTorch sessions.
- **Models**: History tables mapping the final metrics and overall model fitness.
- **Detection**: Target inspection screens to evaluate specific images or videos using saved model versions.

For a comprehensive layout of the architecture, please consult the [DOCUMENTATION.md](../DOCUMENTATION.md).

## FAQ

Frequented asked question documentation can be found in [FAQ.md](FAQ.md).
