# Crack Detection Laboratory

A vintage industrial-style web application for training and deploying crack detection models using YOLO and RF-DETR architectures.

![Version](https://img.shields.io/badge/version-1.0.0-orange)
![License](https://img.shields.io/badge/license-MIT-green)

## Features

- 🎨 **Vintage Industrial UI**: 1970s oscilloscope aesthetic with skeuomorphic design
- 🤖 **Dual Model Support**: Train with YOLO (segmentation) or RF-DETR (detection)
- 🔄 **Real-time Updates**: WebSocket-powered live training metrics and GPU monitoring
- 🎮 **3D Visualization**: Three.js-powered 3D specimen viewer for crack detection results
- 📊 **Live Monitoring**: Real-time GPU stats, temperature, and utilization
- 📁 **Dataset Management**: Import local YOLO-format datasets
- 🧠 **In-Memory Storage**: No database required, simple and fast

## Tech Stack

### Backend
- **FastAPI**: Modern, fast Python web framework
- **Ultralytics YOLO**: State-of-the-art object detection and segmentation
- **PyTorch**: Deep learning framework (supports NVIDIA CUDA and Mac M1/M2/M3)
- **WebSocket**: Real-time bidirectional communication
- **pynvml**: NVIDIA GPU monitoring (Linux/Windows only)

### Frontend
- **React 18**: Modern UI library
- **TypeScript**: Type-safe development
- **Vite**: Next-generation frontend tooling
- **Bun**: Fast JavaScript runtime and package manager
- **Three.js**: 3D graphics library
- **React Three Fiber**: React renderer for Three.js
- **Tailwind CSS**: Utility-first CSS framework
- **Socket.io-client**: WebSocket client

## Quick Start

### Prerequisites
- Python 3.10+ (3.13+ supported with latest PyTorch)
- Bun 1.0+
- **For training:** 
  - NVIDIA GPU with CUDA (Linux/Windows)
  - Mac M1/M2/M3 (CPU training only)

### Backend Setup

#### 1. Create Virtual Environment

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

#### 2. Install Dependencies

```bash
# Install all dependencies (includes PyTorch)
pip install -r requirements.txt
```

**Note:** If you're using Python 3.13+, the latest PyTorch will be installed automatically. If you need specific PyTorch versions for CUDA support, install torch separately first:

**For NVIDIA GPU with CUDA 11.8:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

**For NVIDIA GPU with CUDA 12.1:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

**Note for Mac users:** The `pynvml` package (NVIDIA GPU monitoring) is not compatible with Mac. If you encounter errors, install dependencies without it:
```bash
pip install fastapi uvicorn python-socketio python-multipart ultralytics torch torchvision numpy pillow opencv-python pydantic pyyaml python-dateutil
```

#### 4. Configure Environment

```bash
# Copy environment file
cp .env.example .env

# Edit .env file if needed
```

#### 5. Run the Server

```bash
python main.py
```

The backend will start on `http://localhost:8000`

**Backend Requirements:**
- Python 3.10+
- PyTorch 2.1.2
- See `backend/requirements.txt` for full list
- NVIDIA GPU with CUDA (optional, for GPU-accelerated training)
- Mac M1/M2/M3 users: Training will use CPU (PyTorch MPS support not included in this version)

### Frontend Setup

```bash
cd frontend

# Install dependencies
bun install

# Copy environment file
cp .env.example .env

# Run development server
bun run dev
```

The frontend will start on `http://localhost:5173`

### Docker Setup (Alternative)

```bash
# Build and run with docker-compose
docker-compose up --build

# Backend: http://localhost:8000
# Frontend: http://localhost:5173
```

## Usage

### 1. Import a Dataset
1. Go to **Datasets** page
2. Enter the path to your YOLO-format dataset folder
3. Click **Import**

Dataset structure should be:
```
dataset_folder/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── data.yaml
```

### 2. Train a Model
1. Go to **Training** page
2. Select model type (YOLO or RF-DETR)
3. Choose your imported dataset
4. Configure hyperparameters (epochs, batch size, etc.)
5. Click **Start Training**
6. Monitor live metrics via WebSocket

### 3. Run Prediction
1. Go to **Prediction** page
2. Select a trained model
3. Enter the path to your test image
4. Click **Run Prediction**
5. View results in 3D visualization

## API Endpoints

### Datasets
- `GET /api/datasets` - List all datasets
- `POST /api/datasets/import` - Import dataset from path
- `DELETE /api/datasets/{id}` - Delete dataset

### Training
- `POST /api/training/start` - Start training session
- `POST /api/training/{id}/stop` - Stop training
- `GET /api/training/{id}/status` - Get training status
- `GET /api/training/{id}/metrics` - Get training metrics
- `GET /api/training/sessions` - List all sessions

### Prediction
- `POST /api/prediction` - Run prediction
- `GET /api/prediction/{id}` - Get prediction result

### Models
- `GET /api/models` - List trained models
- `GET /api/models/{id}` - Get model details

### System
- `GET /api/system/gpu` - Get GPU stats
- `GET /api/system/health` - Health check

### WebSocket
- `ws://localhost:8000/ws/training/{id}` - Training live updates
- `ws://localhost:8000/ws/system` - System/GPU updates

## Project Structure

```
.
├── backend/
│   ├── main.py              # FastAPI application
│   ├── training.py          # Training functions
│   ├── prediction.py        # Prediction functions
│   ├── gpu_monitor.py       # GPU monitoring
│   ├── storage.py           # In-memory storage
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/
│   ├── src/
│   │   ├── api.ts           # API client
│   │   ├── App.tsx          # Main application
│   │   ├── main.tsx         # Entry point
│   │   ├── index.css        # Styles with skeuomorphic design
│   │   ├── components/
│   │   │   ├── ui/          # UI components (LED, Button, Panel)
│   │   │   └── layout/      # Layout components
│   │   ├── hooks/
│   │   │   └── useWebSocket.ts
│   │   └── pages/
│   │       ├── Dashboard.tsx
│   │       ├── Training.tsx
│   │       ├── Prediction.tsx
│   │       ├── Datasets.tsx
│   │       ├── Models.tsx
│   │       └── Settings.tsx
│   ├── package.json
│   ├── tailwind.config.js
│   └── Dockerfile
│
├── docker-compose.yml
└── README.md
```

## Design Philosophy

This application features a **vintage industrial laboratory aesthetic** inspired by 1970s oscilloscopes and test equipment:

- **Dark theme** with charcoal background (#1A1A1A)
- **Industrial orange** accents (#FF6B35)
- **CRT green** data displays (#00FF41)
- **Skeuomorphic elements**: Metal panels, 3D buttons, LED indicators, analog gauges
- **Three.js integration**: 3D specimen visualization with vintage lab lighting

## Configuration

### Backend (.env)
```
APP_NAME=Crack Detection API
DEBUG=true
HOST=0.0.0.0
PORT=8000
CORS_ORIGINS=http://localhost:5173
CHECKPOINT_DIR=./checkpoints
DATASET_DIR=./datasets
```

### Frontend (.env)
```
VITE_API_URL=http://localhost:8000/api
VITE_WS_URL=ws://localhost:8000
```

## Development

### Backend Development
```bash
cd backend
python main.py
```

### Frontend Development
```bash
cd frontend
bun run dev
```

### Build for Production
```bash
cd frontend
bun run build
```

## Notes

- **In-Memory Storage**: Data is lost when the server restarts. For production, consider adding a database.
- **GPU Support**: 
  - **NVIDIA GPU**: Full support with CUDA acceleration for training
  - **Mac M1/M2/M3**: CPU training only (no GPU acceleration)
- **Mac Limitations**: GPU monitoring (pynvml) is not available on Mac systems
- **RF-DETR**: Currently placeholder implementation. Full support coming soon.

## License

MIT License - feel free to use this project for your own crack detection applications!

## Acknowledgments

- UI Design inspired by vintage Tektronix oscilloscopes
- Built with love for the crack detection community
