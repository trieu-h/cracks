# Crack Detection Laboratory - Complete Documentation

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Installation & Setup](#installation--setup)
4. [User Guide](#user-guide)
5. [API Reference](#api-reference)
6. [Configuration](#configuration)
7. [Advanced Usage](#advanced-usage)
8. [Troubleshooting](#troubleshooting)
9. [Development Guide](#development-guide)

---

## Overview

### What is This Project?

This is a comprehensive web application designed with a vintage industrial, 1970s oscilloscope aesthetic, built for training, tracking, and deploying YOLO-based AI models specifically mapped for **crack detection**. The system provides:

- **Unified Training Pipeline**: Strictly utilizes the highly-efficient YOLO architecture (standardized on YOLOv26/YOLO11) for accurate crack segmentation.
- **Robust Inference**: Real-time evaluation support for images, videos, and live feeds with adjustable settings.
- **Persistent Training History**: Built-in SQLite database tracking all sessions natively with seamless resumption functionality.
- **Offline Run Syncing**: Dynamically monitors and imports trained model metrics externally dropped in the checkpoints directory.
- **Live System Monitoring**: WebSocket-powered live insights tracking GPU utilization, active temperatures, and deep learning metrics.

### Key Technologies

- **Backend**: Python 3.10+, FastAPI, SQLite, SQLite3 standard library
- **Frontend**: React 18, TypeScript, Vite, Bun, Tailwind CSS
- **ML Framework**: Ultralytics YOLOv11/YOLOv26, PyTorch
- **Live Communication**: WebSocket (via FastAPI and socket.io-client)
- **Monitoring**: pynvml (NVIDIA GPU API integration)


---

## Architecture

### System Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                       React 18 Frontend                     │
│               (Vite, Bun, Tailwind, Three.js)               │
└───────────────────────┬─────────────────────────────────────┘
                        │ HTTP / WebSocket APIs
        ┌───────────────┼───────────────┐
        │               │               │
┌───────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐
│  main.py     │ │ training.py │ │ detection.py│
│ (FastAPI App)│ │ (Pipelines) │ │ (Inference) │
└───────┬──────┘ └──────┬───────┘ └──────┬──────┘
        │               │               │
        └───────────────┼───────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
┌───────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐
│  database.py │ │ storage.py  │ │ sync.py     │
│ (SQLite DB)  │ │ (In-Memory) │ │ (Watcher)   │
└──────────────┘ └─────────────┘ └─────────────┘
                        │
        ┌───────────────▼───────────────┐
        │      Ultralytics Framework    │
        │          (YOLOv26/v11)        │
        └───────────────────────────────┘
```

### Component Overview

#### Backend Modules (`backend/`)

1. **main.py**
   - Main FastAPI application routing and WebSocket endpoints.
   - Bootstraps background services (DB initiation, storage preload, GPU monitor).

2. **database.py**
   - SQLite integration (`data/app.db`) mapping dataset records.
   - Houses persistent tracking for multi-epoch training sessions and configurations.

3. **training.py**
   - Initiates asynchronous training threads natively communicating through WebSockets.
   - Capable of resuming interrupted YOLO checkpoints securely.

4. **detection.py & video_detection.py**
   - Houses inference loops evaluating models against uploaded media.
   - Supports segmented predictions appending dynamic visual mappings for cracks.

5. **sync.py**
   - The offline observer fetching externally completed Ultralytics metric configurations.
   - Directly maps `results.csv` and config metrics into to the front-facing history table.

6. **gpu_monitor.py**
   - Bridges the gap with hardware by extracting direct `pynvml` metrics (temperatures, memory, utilization).

#### Frontend Components (`frontend/`)

1. **pages/**
   - **Dashboard.tsx**: Heads-up telemetry, live GPU/Server readings, and recent session overviews.
   - **Training.tsx**: The central command interface plotting charts via websocket metrics and managing execution lifecycle.
   - **Detection.tsx**: Dedicated file/video inspection terminal.
   - **Datasets.tsx**: Directory ingestion interface strictly capturing `.yaml` mappings.
   - **Models.tsx**: Checkpoint gallery and synchronization control layout.

---

## Installation & Setup

### System Requirements

- **OS**: Windows 10+ / Linux (Ubuntu recommended for pynvml features) / macOS.
- **Node**: Bun 1.0+
- **Python**: 3.10+
- **GPU**: NVIDIA GPU with CUDA strongly recommended for training components.

### Step-by-Step Installation

#### 1. Backend Preparation

1. **Create Virtual Environment**:
```bash
cd backend
python -m venv venv
```

2. **Activate Environment**:
   - **Windows**: `venv\Scripts\activate`
   - **Linux/Mac**: `source venv/bin/activate`

3. **Install PyTorch (Prioritized for CUDA)**:
```bash
# Example for CUDA 11.8 (Check PyTorch site for your specific version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

4. **Install Remaining Packages**:
```bash
pip install -r requirements.txt
```

#### 2. Frontend Preparation

1. **Navigate and Install**:
```bash
cd ../frontend
bun install
```

---

## User Guide

### 1. Dataset Integration

**Purpose**: Linking local dataset pathways to the laboratory database.

**Usage**:
1. Open the UI, click **Datasets**.
2. Supply a root directory path structured consistently for YOLO training standard:
   ```text
   custom_dataset/
   ├── train/
   │   ├── images/
   │   └── labels/
   ├── val/
   │   ├── images/
   │   └── labels/
   └── data.yaml
   ```
3. Initialize the parsing step. The system ingests class maps, sample sizes, and commits this securely into the local `app.db`.

### 2. Training Center

**Purpose**: Configure, launch, and monitor Deep Learning metrics dynamically.

**Features**:
- **Dataset Linking**: Direct ingestion of previously linked profiles.
- **Model Standard**: Uniquely handles YOLO segment architectures.
- **Session Continuation**: Fully supports isolated **resume checks**—if training crashes midway, simply resume it directly from its last known epoch. 

**Execution Cycle**:
1. Assign basic parameters (batch sizes, workers, epochs).
2. Hit start. Observe live chart progression capturing `loss` indicators immediately transmitted natively across WebSockets.
3. Successful completion cascades latest parameters directly into the dashboard.

### 3. Evaluation & Inference

**Purpose**: Test models securely mapped within the environment against varied target specimens.

**Features**:
- Evaluates `.jpg, .png, .mp4`.
- Dynamic confidence sliders (optional configurations based on recent UI adjustments logic).


---

## API Reference

### Dataset Management
- `GET /api/datasets` - Array listing all linked datasets.
- `POST /api/datasets/import` - Expects `{ "path": str }`, discovers and reads nested `yaml`.
- `DELETE /api/datasets/{dataset_id}` - Detach relationship via backend deletion.

### Training Control
- `POST /api/training/start` - Creates thread, issues unique Session ID.
- `POST /api/training/{session_id}/stop` - Halt background task.
- `POST /api/training/{session_id}/resume` - Recovers checkpoint, triggers continuity.
- `GET /api/training/sessions` - Invokes sync mechanism checking locally dropped folders, then responds.

### Model Inspection
- `GET /api/models` - Lists all `.pt` checkpoints inside `/checkpoints` and respective `runs/`.

### Deep Prediction
- `POST /api/detection/upload` - Expects `multipart/form-data`, returns processed segmentation mapping ID.
- `POST /api/detection/video` - Asynchronous background evaluation against moving targets.

---

## Configuration

**Backend Settings `.env`**

```env
APP_NAME=Crack Detection API
DEBUG=true
HOST=0.0.0.0
PORT=8000
CORS_ORIGINS=http://localhost:5173
CHECKPOINT_DIR=./checkpoints
DATASET_DIR=./datasets
```

**Frontend Settings `.env`**

```env
VITE_API_URL=http://localhost:8000/api
VITE_WS_URL=ws://localhost:8000
```

---

## Advanced Usage

### Sync Mechanism (Offline Loading)

The codebase supports `discover_offline_runs()` natively injected into `/api/training/sessions`.
If you or team members decide to execute a complex model externally remotely and simply "drop" the executed `train/runs/...` output into your local `checkpoints` folder:
- The system automatically parses the `args.yaml` and `results.csv`.
- Retrospectively structures the history dashboard mapping F1 Scores, iterations seamlessly without a true backend session origin.

### Skeleton Model Defaults 
Currently built to support variations like:
- `yolo26n.pt`
- `yolo11n-seg.pt`

By default, the UI prioritizes segmentation variant inference to encapsulate cracks via mask boundaries instead of merely drawing boxes.

---

## Troubleshooting

### 1. File Path Resolution Error during Database Sync
**Problem**: Submitting dataset paths raises failure regarding missing YAML.
**Solution**: Normalize windows paths mapping correctly, ensuring absolute paths vs relative context are identical natively to where the FastAPI wrapper initiates.

### 2. WebSocket Failure during Training
**Problem**: The UI begins executing the model but metrics flatline.
**Solution**: Usually occurs when a training port crashes out of bounds from `CORS_ORIGINS`. Verify your exact port mappings on both `.env` layers align accurately. Also ensure PyTorch CUDA memory isn't exhausting background threads unnoticeably (can be checked in backend `error.txt`).

### 3. Mac M-Series Environments
**Problem**: Terminal throws pynvml exceptions on macOS launch.
**Solution**: The macOS architecture naturally rejects `pynvml` packages since there are no physical NVIDIA cores. Code execution defaults to CPU safely, but warning traces can be suppressed by modifying `monitor_gpu` within `gpu_monitor.py`.

---

## Development Guide

### Adding an Analytical Feature

When extending dataset variables or historical metrics, ensure synchronicity strictly between:
1. `backend/database.py` DB creation schema fields.
2. The UI table structure located in the `Training` or `Models` view.
3. The offline parser configuration wrapped in `sync.py`.

**Documentation Version**: 1.1.0  
**Last Updated**: 2026-04-15
