# Crack Detection Laboratory

A comprehensive web application featuring a vintage industrial aesthetic for training, tracking, and deploying YOLO segmentation models to detect structural cracks in physical materials. 

## 🚀 Features

### 📊 Dashboard
- **Real-time Metrics**: Monitor model performance and general tracking from your latest training iterations.
- **System Telemetry**: Live hardware monitoring including GPU memory and temperature.
- **Recent Sessions**: View a quick gallery of previous model checkpoints.

### 🔮 Prediction
- **Image & Video Processing**: Upload media and receive instant crack detection with precise segmentation overlays.
- **Live Camera**: Real-time evaluation support for live hardware feeds.
- **Model Selection**: Switch between multiple YOLO architectures directly inside the UI.

### 🏋️ Training
- **Dataset Management**: Centralized import interface for parsing YOLO format structural datasets.
- **Hyperparameter Configuration**: Flexible configuration with adjustable parameters designed for immediate hardware limits.
- **Live Training Progress**: Visual charts powered by WebSockets to monitor training loss in real time.
- **Session Resumption**: Automatically resume interrupted model checkpoints from exactly where a crash occurred.

## 📁 Project Structure

```text
crack-detection-ui/
├── backend/                    # Python FastAPI application and PyTorch pipelines
├── frontend/                   # React 18 / Vite frontend dashboard
├── docs/                       # Project Documentation
└── README.md                   # This file
```

## 🛠️ Installation

### Prerequisites
- Python 3.10 or higher
- Bun 1.0+ (for frontend packages)
- An active NVIDIA GPU (with CUDA) is strongly recommended to run PyTorch training efficiently.

### Setup Instructions

1. **Clone the repository** and open the folder.
2. **Backend Setup**:
```bash
cd backend
python -m venv venv

# Windows Users:
venv\Scripts\activate
# Mac/Linux Users:
source venv/bin/activate

# Install PyTorch and dependencies:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```
3. **Frontend Setup**:
```bash
cd ../frontend
bun install
```

## 🎯 Usage

1. **Start the Backend API**:
```bash
cd backend
python main.py
```

2. **Start the Frontend UI**:
```bash
cd frontend
bun run dev
```
Open your web browser and navigate to `http://localhost:5173`. 

### Quick Start Guide
1. **Link Dataset**: Use the **Datasets** module to point the app to your crack dataset directory.
2. **Launch Routine**: Navigate to **Training**, select your image set, and begin the run.
3. **Test Segments**: Swap to the **Detection** tab, upload a test image, and generate segmented output.

> [!TIP]
> For a more detailed step-by-step guide on navigating the UI, review the [WORKFLOW.md](docs/WORKFLOW.md) document.

## 📖 Complete Documentation
Detailed manuals regarding the architecture, parameters, and sync mechanisms can be found in [DOCUMENTATION.md](DOCUMENTATION.md).

## 🤝 Support & License
This project operates under a strict **End-User License Agreement (EULA)** that permits local usage, but strictly prohibits unapproved distributions or derivative modifications. Please review the [LICENSE](LICENSE) file for exact boundaries. Acknowledgment provided to Ultralytics.
