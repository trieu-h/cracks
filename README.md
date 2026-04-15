# Crack Detection Laboratory

A modern, comprehensive web application featuring a vintage industrial aesthetic for training and deploying YOLOv26 segmentation models to detect structural cracks in materials. This project provides a complete workflow from dataset preparation to model training, evaluation, and real-time inference using WebSocket metrics.

## 🚀 Features

### 📊 Dashboard
- **Real-time Metrics**: View model performance metrics and dataset tracking from latest training iterations.
- **System Telemetry**: Live GPU monitoring (utilization, memory, temperature) and background server health.
- **Recent Sessions**: Gallery of previous checkpoints and synchronized runs.

### 🔮 Prediction
- **Image & Video Processing**: Upload media and get instant crack detection with precise segmentation overlays.
- **Live Camera**: Real-time evaluation support for live feeds with adjustable confidence threshold settings.
- **3D Visualization**: Unique Three.js integration simulating cracks on 3D rendered specimens in an industrial lab context.
- **Model Selection**: Choose between YOLO segmentation architectures natively inside the laboratory.

### 🏋️ Training
- **Dataset Management**: 
  - Centralized import interface parsing YOLO format structural datasets.
  - Integration securely tracked in a native SQLite database.
- **Hyperparameter Configuration**: Flexible configuration with batch size, worker allocations, and direct mapping options.
- **Live Training Progress**:
  - WebSockets transmitting loss charts and progress tracking natively in real-time.
  - Native checkpoint resumption capabilities mapping directly from the exact epoch a crash/termination occurred.
- **Run Syncing**: Auto-discover and parse metric logs from external `train/runs` dropped into the main `/checkpoints` hub.
- **Complete Monitoring**: pynvml NVIDIA GPU API integration feeding live telemetry without performance overhead.

## 📁 Project Structure

```
crack-detection-ui/
├── backend/                    # Python FastAPI Backend
│   ├── main.py                 # FastAPI application & WebSockets
│   ├── training.py             # Asynchronous PyTorch pipelines
│   ├── detection.py            # Inference and Media Evaluation
│   ├── database.py             # SQLite persistence tables
│   ├── sync.py                 # Offline checkpoint auto-discovery
│   └── requirements.txt        # Backend dependencies
│
├── frontend/                   # React 18 / Vite Frontend
│   ├── src/
│   │   ├── App.tsx             # Application router
│   │   ├── pages/              # Views (Dashboard, Training, etc.)
│   │   └── components/         # Skeuomorphic UI Elements
│   ├── package.json            # Frontend dependencies
│   └── tailwind.config.js      # Styling framework mapping
│
├── docs/                       # Project Documentation
│   ├── DOCUMENTATION.md        # Complete framework documentation
│   └── IMPLEMENTATION_PLAN.md  # Roadmap
│
└── README.md                   # This file
```

## 🛠️ Installation

### Prerequisites
- Python 3.10 or higher
- Bun 1.0+ (for frontend)
- NVIDIA GPU with CUDA strongly recommended for PyTorch.

### Step-by-Step Setup

1. **Clone or navigate to the repository directory**
2. **Backend Setup**:
```bash
cd backend
python -m venv venv

# Activate Environment (Windows)
venv\Scripts\activate
# Activate Environment (Mac/Linux)
source venv/bin/activate

# Install Priority PyTorch with CUDA 11.8+
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```
3. **Frontend Setup**:
```bash
cd ../frontend
bun install
```
4. **Configuration**: Set your `.env` variables in both frontend/backend directories as per `DOCUMENTATION.md` outlines.

## 🎯 Usage

### Starting the Laboratory

1. **Backend Initialize**:
```bash
cd backend
python main.py
```

2. **Frontend Launch**:
```bash
cd frontend
bun run dev
```
The application will open natively via your web browser (typically `http://localhost:5173`) mapping to the backend API via `http://localhost:8000/api`.

### Quick Start Guide
1. **Link Dataset**: Use the **Datasets** module to ingest your local structural/crack dataset YAML.
2. **Launch Routine**: Navigate to **Training**, link previous session or standard YOLOv26 node, and begin. Monitor precise PyTorch progress mapped straight over WebSockets.
3. **Test Segments**: Swap to **Detection**, attach your video or image file, and visualize the cracks either directly or projected over 3D simulation specimens.

## 📋 Requirements
- **FastAPI** (Backend framework)
- **Ultralytics** (Deep Learning architecture base)
- **PyTorch** 
- **Three.js / React-Three-Fiber** (3D Mapping Integration)
- **pynvml** (NVIDIA Metrics parsing)

## 📖 Documentation
Detailed documentation including API reference, configurations, offline-sync mechanism, and setup is housed under:
[docs/DOCUMENTATION.md](docs/DOCUMENTATION.md)

## 🐛 Troubleshooting

1. **CUDA Mismatch**: Ensure system PyTorch version matches physical NVIDIA driver capability. Check natively using `nvidia-smi`.
2. **Mac Limitations**: Pynvml library throws exceptions natively over Apple Silicon (M1/M2/M3) as no Nvidia GPU exists.
3. **WebSocket Loss**: Check terminal window tracing if PyTorch exhausted VRAM resulting in an abrupt thread termination.

## 🤝 Support & License

This project implements standard MIT licensing logic frameworks. Acknowledgment provided to Ultralytics.

---
**Version**: 1.0.0
**Last Updated**: 2026-04-15
