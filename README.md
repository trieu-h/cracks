# Crack Detection Laboratory

Welcome to the Crack Detection Laboratory! This is a simple, easy-to-use application designed to look like a vintage 1970s laboratory. It helps you train an Artificial Intelligence (AI) to automatically find cracks in buildings, bridges, and machines.

## 🚀 Features

### 📊 Dashboard
- **Live Numbers**: Watch the AI learn in real-time.
- **Computer Health**: See how hard your computer is working (checks temperature and memory).
- **History**: Look back at all the AI models you have trained in the past.

### 🔮 Finding Cracks (Prediction)
- **Pictures & Videos**: Upload your files and the AI will color over the cracks instantly!
- **Live Camera**: Connect a camera to see the AI find cracks in real-time.
- **Model Choice**: Switch between different AI brains you have trained.

### 🏋️ Teaching the AI (Training)
- **Manage Pictures**: Easily group and save folders of crack pictures.
- **Simple Settings**: Change how fast or how detailed you want the AI to learn.
- **Live Tracking**: See simple charts that show you exactly how smart the AI is getting.
- **Safety Resume**: If your computer turns off by mistake, the AI remembers everything and can continue exactly where it left off!

## 📁 How the Files are Organized

```text
crack-detection-ui/
├── backend/                    # The brain behind the app (Python)
├── frontend/                   # What you see on the screen (React)
├── docs/                       # Help guides and manuals
└── README.md                   # This file!
```

## 🛠️ How to Install

### What you need:
- **Python** (version 3.10 or higher)
- **Bun** (version 1.0 or higher)
- **A modern computer** (An NVIDIA graphics card is recommended for faster learning).

### Setup Steps:

1. **Download this folder** to your computer.
2. **Setup the Brain (Backend)**:
```bash
cd backend
python -m venv venv

# Windows Users:
venv\Scripts\activate
# Mac/Linux Users:
source venv/bin/activate

# Install the necessary files:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```
3. **Setup the Screen (Frontend)**:
```bash
cd ../frontend
bun install
```

## 🎯 How to Start the App

1. **Start the Brain**:
```bash
cd backend
python main.py
```

2. **Start the Screen**:
```bash
cd frontend
bun run dev
```
Open your web browser and go to `http://localhost:5173`. 

### Quick Start Guide
1. **Add Pictures**: Use the **Datasets** tab to add a folder of crack pictures.
2. **Teach the AI**: Go to the **Training** tab, select your pictures, and click start.
3. **Test It**: Go to the **Detection** tab, upload a video, and watch the AI find the cracks!

> [!TIP]
> For a more detailed step-by-step guide on how to use the app, check the [WORKFLOW.md](docs/WORKFLOW.md) file.

## 📖 Help & Guides
If you want to read more about how the app works, please read the [DOCUMENTATION.md](DOCUMENTATION.md).

## 🤝 Rules of Use
This project is private. Please review the [LICENSE](LICENSE) file to see the rules about using this software.
