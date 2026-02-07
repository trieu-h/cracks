# Crack Detection Training Client - Implementation Plan

## Executive Summary

A full-stack application for training and deploying crack detection models using YOLO (v8-seg, v9-seg) and RF-DETR architectures with Roboflow dataset integration and real-time GPU monitoring.

---

## 1. Project Structure

```
crack-detection-client/
├── README.md
├── docker-compose.yml
├── .env.example
├── .gitignore
├── Makefile
│
├── backend/                          # Python FastAPI Backend
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                   # FastAPI application entry
│   │   ├── config.py                 # Pydantic settings
│   │   │
│   │   ├── api/                      # API Routes
│   │   │   ├── __init__.py
│   │   │   ├── v1/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── router.py
│   │   │   │   ├── endpoints/
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── datasets.py
│   │   │   │   │   ├── training.py
│   │   │   │   │   ├── prediction.py
│   │   │   │   │   ├── dashboard.py
│   │   │   │   │   ├── models.py
│   │   │   │   │   └── system.py
│   │   │   │   └── websockets/
│   │   │   │       ├── __init__.py
│   │   │   │       ├── training_ws.py
│   │   │   │       └── system_ws.py
│   │   │
│   │   ├── core/                     # Core Business Logic
│   │   │   ├── __init__.py
│   │   │   ├── training/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── trainer.py
│   │   │   │   ├── yolo_trainer.py
│   │   │   │   ├── rfdetr_trainer.py
│   │   │   │   ├── callbacks.py
│   │   │   │   └── config.py
│   │   │   ├── prediction/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── predictor.py
│   │   │   │   ├── yolo_predictor.py
│   │   │   │   └── rfdetr_predictor.py
│   │   │   └── monitoring/
│   │   │       ├── __init__.py
│   │   │       ├── gpu_monitor.py
│   │   │       └── metrics_collector.py
│   │   │
│   │   ├── data/                     # Data Pipeline
│   │   │   ├── __init__.py
│   │   │   ├── roboflow_loader.py
│   │   │   ├── preprocessor.py
│   │   │   ├── augmentor.py
│   │   │   └── validators.py
│   │   │
│   │   ├── models/                   # Database Models
│   │   │   ├── __init__.py
│   │   │   ├── database.py
│   │   │   ├── training_session.py
│   │   │   ├── dataset.py
│   │   │   ├── checkpoint.py
│   │   │   └── prediction.py
│   │   │
│   │   ├── schemas/                  # Pydantic Schemas
│   │   │   ├── __init__.py
│   │   │   ├── common.py
│   │   │   ├── dataset.py
│   │   │   ├── training.py
│   │   │   ├── prediction.py
│   │   │   ├── dashboard.py
│   │   │   └── system.py
│   │   │
│   │   └── utils/                    # Utilities
│   │       ├── __init__.py
│   │       ├── filesystem.py
│   │       ├── validators.py
│   │       └── logger.py
│   │
│   ├── models/                       # Model Storage (gitignored)
│   │   ├── .gitkeep
│   │   └── README.md
│   │
│   ├── datasets/                     # Dataset Storage (gitignored)
│   │   ├── .gitkeep
│   │   └── README.md
│   │
│   ├── checkpoints/                  # Training Checkpoints (gitignored)
│   │   ├── .gitkeep
│   │   └── README.md
│   │
│   ├── tests/                        # Backend Tests
│   │   ├── __init__.py
│   │   ├── conftest.py
│   │   ├── unit/
│   │   │   ├── __init__.py
│   │   │   ├── test_data_pipeline.py
│   │   │   └── test_gpu_monitor.py
│   │   └── integration/
│   │       ├── __init__.py
│   │       └── test_api.py
│   │
│   ├── alembic/                      # Database Migrations
│   │   ├── env.py
│   │   ├── script.py.mako
│   │   └── versions/
│   │
│   ├── requirements.txt
│   ├── requirements-dev.txt
│   ├── Dockerfile
│   └── pyproject.toml
│
├── frontend/                         # React + TypeScript Frontend
│   ├── public/
│   │   ├── index.html
│   │   ├── favicon.ico
│   │   └── manifest.json
│   │
│   ├── src/
│   │   ├── index.tsx
│   │   ├── App.tsx
│   │   ├── App.css
│   │   ├── react-app-env.d.ts
│   │   ├── setupTests.ts
│   │   │
│   │   ├── components/               # Reusable UI Components
│   │   │   ├── common/
│   │   │   │   ├── Button/
│   │   │   │   │   ├── index.tsx
│   │   │   │   │   └── Button.css
│   │   │   │   ├── Card/
│   │   │   │   ├── Modal/
│   │   │   │   ├── Toast/
│   │   │   │   ├── Loader/
│   │   │   │   ├── ProgressBar/
│   │   │   │   └── ErrorBoundary/
│   │   │   │
│   │   │   ├── layout/
│   │   │   │   ├── Sidebar/
│   │   │   │   ├── Header/
│   │   │   │   ├── Footer/
│   │   │   │   └── MainLayout/
│   │   │   │
│   │   │   ├── charts/
│   │   │   │   ├── LineChart/
│   │   │   │   ├── BarChart/
│   │   │   │   ├── GPUMetricChart/
│   │   │   │   └── LossCurveChart/
│   │   │   │
│   │   │   └── gpu/
│   │   │       ├── GPUMonitor/
│   │   │       ├── VRAMGauge/
│   │   │       ├── TemperatureGauge/
│   │   │       └── UtilizationBar/
│   │   │
│   │   ├── pages/                    # Page Components
│   │   │   ├── Dashboard/
│   │   │   │   ├── index.tsx
│   │   │   │   ├── Dashboard.tsx
│   │   │   │   └── Dashboard.css
│   │   │   ├── Training/
│   │   │   │   ├── index.tsx
│   │   │   │   ├── Training.tsx
│   │   │   │   ├── TrainingConfig/
│   │   │   │   │   ├── DatasetSelector.tsx
│   │   │   │   │   ├── ModelSelector.tsx
│   │   │   │   │   ├── HyperparameterForm.tsx
│   │   │   │   │   └── TrainingOptions.tsx
│   │   │   │   ├── ActiveTraining/
│   │   │   │   │   ├── TrainingMonitor.tsx
│   │   │   │   │   ├── MetricsDisplay.tsx
│   │   │   │   │   └── CheckpointManager.tsx
│   │   │   │   └── TrainingHistory/
│   │   │   │
│   │   │   ├── Prediction/
│   │   │   │   ├── index.tsx
│   │   │   │   ├── Prediction.tsx
│   │   │   │   ├── ImageUploader.tsx
│   │   │   │   ├── BatchProcessor.tsx
│   │   │   │   ├── ResultsViewer.tsx
│   │   │   │   └── ExportOptions.tsx
│   │   │   │
│   │   │   ├── Datasets/
│   │   │   │   ├── index.tsx
│   │   │   │   ├── DatasetList.tsx
│   │   │   │   ├── DatasetImport.tsx
│   │   │   │   └── DatasetViewer.tsx
│   │   │   │
│   │   │   ├── Models/
│   │   │   │   ├── index.tsx
│   │   │   │   ├── ModelList.tsx
│   │   │   │   └── ModelDetails.tsx
│   │   │   │
│   │   │   └── Settings/
│   │   │       ├── index.tsx
│   │   │       ├── GeneralSettings.tsx
│   │   │       └── GPUSettings.tsx
│   │   │
│   │   ├── hooks/                    # Custom React Hooks
│   │   │   ├── useWebSocket.ts
│   │   │   ├── useTraining.ts
│   │   │   ├── useGPUStats.ts
│   │   │   ├── useDatasets.ts
│   │   │   ├── useModels.ts
│   │   │   └── usePrediction.ts
│   │   │
│   │   ├── services/                 # API Services
│   │   │   ├── api.ts                # Axios instance
│   │   │   ├── datasets.service.ts
│   │   │   ├── training.service.ts
│   │   │   ├── prediction.service.ts
│   │   │   ├── dashboard.service.ts
│   │   │   └── system.service.ts
│   │   │
│   │   ├── store/                    # State Management (Zustand)
│   │   │   ├── index.ts
│   │   │   ├── slices/
│   │   │   │   ├── trainingStore.ts
│   │   │   │   ├── gpuStore.ts
│   │   │   │   ├── datasetStore.ts
│   │   │   │   └── uiStore.ts
│   │   │   └── middleware/
│   │   │       └── websocketMiddleware.ts
│   │   │
│   │   ├── types/                    # TypeScript Types
│   │   │   ├── index.ts
│   │   │   ├── dataset.types.ts
│   │   │   ├── training.types.ts
│   │   │   ├── prediction.types.ts
│   │   │   ├── dashboard.types.ts
│   │   │   ├── gpu.types.ts
│   │   │   └── api.types.ts
│   │   │
│   │   ├── utils/                    # Utility Functions
│   │   │   ├── formatters.ts
│   │   │   ├── validators.ts
│   │   │   ├── constants.ts
│   │   │   └── helpers.ts
│   │   │
│   │   ├── styles/                   # Global Styles
│   │   │   ├── variables.css
│   │   │   ├── mixins.css
│   │   │   └── global.css
│   │   │
│   │   └── assets/                   # Static Assets
│   │       ├── images/
│   │       └── icons/
│   │
│   ├── tests/                        # Frontend Tests
│   │   ├── unit/
│   │   ├── integration/
│   │   └── e2e/
│   │
│   ├── package.json
│   ├── tsconfig.json
│   ├── package-lock.json
│   ├── .eslintrc.json
│   ├── .prettierrc
│   └── Dockerfile
│
├── docs/                             # Documentation
│   ├── API.md
│   ├── DEPLOYMENT.md
│   ├── MODELS.md
│   └── ARCHITECTURE.md
│
└── scripts/                          # Utility Scripts
    ├── setup.sh
    ├── dev-start.sh
    └── build.sh
```

---

## 2. Frontend Architecture

### 2.1 Technology Stack

| Category | Technology | Version | Purpose |
|----------|-----------|---------|---------|
| Framework | React | ^18.2.0 | UI Library |
| Language | TypeScript | ^5.0.0 | Type Safety |
| Build Tool | Vite | ^5.0.0 | Fast bundling |
| Styling | Tailwind CSS | ^3.4.0 | Utility-first CSS |
| State Management | Zustand | ^4.4.0 | Lightweight state |
| Data Fetching | TanStack Query | ^5.0.0 | Server state |
| Charts | Recharts | ^2.10.0 | Data visualization |
| UI Components | Radix UI | ^1.0.0 | Headless primitives |
| Icons | Lucide React | ^0.300.0 | Icon library |
| WebSocket | Socket.io-client | ^4.7.0 | Real-time comms |
| HTTP Client | Axios | ^1.6.0 | API requests |
| Testing | Vitest + RTL | ^1.0.0 | Unit tests |

### 2.2 Component Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      App Component                          │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────┐   │
│  │                 MainLayout                           │   │
│  │  ┌────────────┐  ┌─────────────────────────────┐   │   │
│  │  │  Sidebar   │  │        Content Area          │   │   │
│  │  │  - Nav     │  │  ┌────────────────────────┐  │   │   │
│  │  │  - Links   │  │  │    Page Component      │  │   │   │
│  │  │  - Status  │  │  │  (Dashboard/Training/  │  │   │   │
│  │  └────────────┘  │  │   Prediction/etc.)     │  │   │   │
│  │                  │  │                        │  │   │   │
│  │  ┌────────────┐  │  │  - Feature Components  │  │   │   │
│  │  │   Header   │  │  │  - Data Visualization  │  │   │   │
│  │  │  - GPU     │  │  │  - Forms               │  │   │   │
│  │  │    Stats   │  │  │  - Tables              │  │   │   │
│  │  └────────────┘  │  └────────────────────────┘  │   │   │
│  └──────────────────┴──────────────────────────────┘   │   │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 State Management (Zustand)

```typescript
// store/slices/trainingStore.ts
interface TrainingState {
  // Training Session State
  activeSession: TrainingSession | null;
  sessions: TrainingSession[];
  isTraining: boolean;
  
  // Training Metrics
  metrics: TrainingMetrics;
  epochs: EpochData[];
  currentEpoch: number;
  totalEpochs: number;
  
  // Actions
  startTraining: (config: TrainingConfig) => Promise<void>;
  stopTraining: () => Promise<void>;
  updateMetrics: (metrics: Partial<TrainingMetrics>) => void;
  addEpoch: (epoch: EpochData) => void;
  loadSessions: () => Promise<void>;
}

// store/slices/gpuStore.ts
interface GPUState {
  // GPU Statistics
  stats: GPUStats;
  history: GPUHistoryPoint[];
  isMonitoring: boolean;
  
  // Actions
  updateStats: (stats: GPUStats) => void;
  startMonitoring: () => void;
  stopMonitoring: () => void;
  clearHistory: () => void;
}

// store/slices/datasetStore.ts
interface DatasetState {
  datasets: Dataset[];
  selectedDataset: Dataset | null;
  isLoading: boolean;
  
  // Actions
  loadDatasets: () => Promise<void>;
  importDataset: (url: string) => Promise<void>;
  selectDataset: (id: string) => void;
  deleteDataset: (id: string) => Promise<void>;
}
```

### 2.4 Routing Structure

```typescript
// App.tsx routing configuration
const router = createBrowserRouter([
  {
    path: '/',
    element: <MainLayout />,
    errorElement: <ErrorPage />,
    children: [
      {
        index: true,
        element: <DashboardPage />,
      },
      {
        path: 'training',
        element: <TrainingPage />,
        children: [
          { path: 'new', element: <TrainingConfig /> },
          { path: 'active', element: <ActiveTraining /> },
          { path: 'history', element: <TrainingHistory /> },
        ],
      },
      {
        path: 'prediction',
        element: <PredictionPage />,
      },
      {
        path: 'datasets',
        element: <DatasetsPage />,
      },
      {
        path: 'models',
        element: <ModelsPage />,
      },
      {
        path: 'settings',
        element: <SettingsPage />,
      },
    ],
  },
]);
```

---

## 3. Backend Architecture

### 3.1 Technology Stack

| Category | Technology | Version | Purpose |
|----------|-----------|---------|---------|
| Framework | FastAPI | ^0.104.0 | API Framework |
| ASGI Server | Uvicorn | ^0.24.0 | HTTP Server |
| WebSocket | FastAPI WebSocket | Built-in | Real-time comms |
| ML Framework | Ultralytics | ^8.0.0 | YOLO training |
| ML Framework | RF-DETR | ^0.1.0 | RF-DETR training |
| Dataset | Roboflow | ^1.1.0 | Dataset loading |
| GPU Monitoring | pynvml | ^11.5.0 | NVIDIA GPU stats |
| Database | SQLite/PostgreSQL | ^3.0/15.0 | Data persistence |
| ORM | SQLAlchemy | ^2.0.0 | Database ORM |
| Migrations | Alembic | ^1.12.0 | DB migrations |
| Validation | Pydantic | ^2.5.0 | Data validation |
| Background | Celery | ^5.3.0 | Async tasks |
| Cache | Redis | ^7.0.0 | Caching/Broker |
| Testing | pytest | ^7.4.0 | Unit tests |
| Monitoring | Prometheus | ^0.19.0 | Metrics export |

### 3.2 FastAPI Application Structure

```python
# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.api.v1.router import api_router
from app.api.v1.websockets.router import ws_router
from app.core.monitoring.gpu_monitor import GPUMonitor
from app.models.database import init_db

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await init_db()
    app.state.gpu_monitor = GPUMonitor()
    await app.state.gpu_monitor.start()
    yield
    # Shutdown
    await app.state.gpu_monitor.stop()

app = FastAPI(
    title="Crack Detection Training API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(api_router, prefix="/api/v1")
app.include_router(ws_router, prefix="/ws")
```

### 3.3 Training Orchestration

```python
# app/core/training/trainer.py (Abstract Base)
from abc import ABC, abstractmethod
from typing import Dict, Any, Callable, Optional
import asyncio
from pathlib import Path

class BaseTrainer(ABC):
    """Abstract base class for all model trainers."""
    
    def __init__(
        self,
        session_id: str,
        config: TrainingConfig,
        progress_callback: Optional[Callable] = None,
        metrics_callback: Optional[Callable] = None
    ):
        self.session_id = session_id
        self.config = config
        self.progress_callback = progress_callback
        self.metrics_callback = metrics_callback
        self.is_running = False
        self.current_epoch = 0
        
    @abstractmethod
    async def train(self) -> Path:
        """Execute training and return path to best model."""
        pass
    
    @abstractmethod
    async def stop(self):
        """Gracefully stop training."""
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Get current training status."""
        pass
    
    def _on_epoch_end(self, epoch: int, metrics: Dict[str, float]):
        """Called at end of each epoch."""
        self.current_epoch = epoch
        if self.metrics_callback:
            asyncio.create_task(
                self.metrics_callback(self.session_id, epoch, metrics)
            )

# app/core/training/yolo_trainer.py
from ultralytics import YOLO
from app.core.training.trainer import BaseTrainer

class YOLOTrainer(BaseTrainer):
    """YOLOv8/v9 segmentation trainer."""
    
    async def train(self) -> Path:
        self.is_running = True
        
        # Load model
        model = YOLO(self.config.model_path)
        
        # Training arguments
        args = {
            'data': self.config.dataset_yaml,
            'epochs': self.config.epochs,
            'imgsz': self.config.image_size,
            'batch': self.config.batch_size,
            'lr0': self.config.learning_rate,
            'lrf': self.config.lr_factor,
            'momentum': self.config.momentum,
            'weight_decay': self.config.weight_decay,
            'patience': self.config.patience,
            'save': True,
            'project': self.config.output_dir,
            'name': self.session_id,
            'exist_ok': True,
            'pretrained': self.config.pretrained,
            'device': self.config.device,
            'workers': self.config.num_workers,
            'cos_lr': self.config.cosine_lr,
            'close_mosaic': self.config.close_mosaic,
            'augment': self.config.use_augmentation,
        }
        
        # Add callbacks
        callbacks = {
            'on_epoch_end': self._on_epoch_end,
            'on_train_end': self._on_train_end,
        }
        
        # Run training
        results = model.train(**args, callbacks=callbacks)
        
        return Path(results.best)

# app/core/training/rfdetr_trainer.py  
from rfdetr import RFDETR
from app.core.training.trainer import BaseTrainer

class RFDETRTrainer(BaseTrainer):
    """RF-DETR object detection trainer."""
    
    async def train(self) -> Path:
        self.is_running = True
        
        model = RFDETR(
            num_classes=self.config.num_classes,
            pretrained=True
        )
        
        history = model.fit(
            dataset=self.config.dataset_yaml,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            callbacks=[self._create_callback()]
        )
        
        return self._save_model(model)
```

### 3.4 GPU Monitoring Service

```python
# app/core/monitoring/gpu_monitor.py
import pynvml
import asyncio
from typing import List, Dict, Any
from datetime import datetime
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class GPUStats:
    index: int
    name: str
    temperature: float
    utilization: float
    memory_used: int
    memory_total: int
    memory_free: int
    power_draw: float
    power_limit: float
    fan_speed: float
    timestamp: datetime

class GPUMonitor:
    """Real-time GPU monitoring using NVML."""
    
    def __init__(self, update_interval: float = 2.0):
        self.update_interval = update_interval
        self._running = False
        self._task = None
        self._subscribers = []
        self._history = []
        self._max_history = 300  # 10 minutes at 2s intervals
        
        # Initialize NVML
        try:
            pynvml.nvmlInit()
            self.device_count = pynvml.nvmlDeviceGetCount()
            self.handles = [
                pynvml.nvmlDeviceGetHandleByIndex(i) 
                for i in range(self.device_count)
            ]
            logger.info(f"GPU Monitor initialized: {self.device_count} devices")
        except pynvml.NVMLError as e:
            logger.error(f"NVML initialization failed: {e}")
            self.device_count = 0
            self.handles = []
    
    async def start(self):
        """Start monitoring loop."""
        if self._running or self.device_count == 0:
            return
        
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info("GPU monitoring started")
    
    async def stop(self):
        """Stop monitoring loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        pynvml.nvmlShutdown()
        logger.info("GPU monitoring stopped")
    
    async def _monitor_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                stats = self._get_all_stats()
                self._history.append(stats)
                
                # Trim history
                if len(self._history) > self._max_history:
                    self._history = self._history[-self._max_history:]
                
                # Notify subscribers
                for callback in self._subscribers:
                    try:
                        await callback(stats)
                    except Exception as e:
                        logger.error(f"Subscriber callback error: {e}")
                
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"GPU monitoring error: {e}")
                await asyncio.sleep(self.update_interval)
    
    def _get_all_stats(self) -> List[GPUStats]:
        """Get stats for all GPUs."""
        stats = []
        timestamp = datetime.utcnow()
        
        for i, handle in enumerate(self.handles):
            try:
                # Temperature
                temp = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
                
                # Utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                
                # Memory
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                # Power
                try:
                    power_draw = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000
                    power_limit = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000
                except pynvml.NVMLError:
                    power_draw = 0
                    power_limit = 0
                
                # Fan speed
                try:
                    fan_speed = pynvml.nvmlDeviceGetFanSpeed(handle)
                except pynvml.NVMLError:
                    fan_speed = 0
                
                # Device name
                name = pynvml.nvmlDeviceGetName(handle)
                
                stats.append(GPUStats(
                    index=i,
                    name=name,
                    temperature=temp,
                    utilization=util.gpu,
                    memory_used=mem_info.used,
                    memory_total=mem_info.total,
                    memory_free=mem_info.free,
                    power_draw=power_draw,
                    power_limit=power_limit,
                    fan_speed=fan_speed,
                    timestamp=timestamp
                ))
                
            except pynvml.NVMLError as e:
                logger.error(f"Error reading GPU {i}: {e}")
        
        return stats
    
    def subscribe(self, callback):
        """Subscribe to GPU stats updates."""
        self._subscribers.append(callback)
    
    def unsubscribe(self, callback):
        """Unsubscribe from GPU stats updates."""
        if callback in self._subscribers:
            self._subscribers.remove(callback)
    
    def get_current_stats(self) -> List[GPUStats]:
        """Get current stats (synchronous)."""
        return self._get_all_stats()
    
    def get_history(self) -> List[List[GPUStats]]:
        """Get historical stats."""
        return self._history
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of GPU information."""
        return {
            'device_count': self.device_count,
            'devices': [
                {
                    'index': i,
                    'name': pynvml.nvmlDeviceGetName(handle)
                }
                for i, handle in enumerate(self.handles)
            ]
        }
```

---

## 4. Data Pipeline

### 4.1 Roboflow Dataset Loader

```python
# app/data/roboflow_loader.py
import yaml
import shutil
from pathlib import Path
from typing import Dict, Optional, List
import zipfile
import requests
from roboflow import Roboflow
import logging

from app.schemas.dataset import DatasetInfo, DatasetSplit

logger = logging.getLogger(__name__)

class RoboflowLoader:
    """Load and manage Roboflow datasets in YOLO format."""
    
    def __init__(self, api_key: Optional[str] = None, workspace: Optional[str] = None):
        self.api_key = api_key
        self.workspace = workspace
        self.rf = Roboflow(api_key=api_key) if api_key else None
        self.datasets_dir = Path("datasets")
        self.datasets_dir.mkdir(exist_ok=True)
    
    async def download_dataset(
        self,
        workspace: str,
        project: str,
        version: int,
        format: str = "yolov8"
    ) -> DatasetInfo:
        """
        Download dataset from Roboflow.
        
        Args:
            workspace: Roboflow workspace name
            project: Project name
            version: Dataset version number
            format: Export format (yolov8, yolov9, coco, etc.)
            
        Returns:
            DatasetInfo with paths and metadata
        """
        if not self.rf:
            raise ValueError("Roboflow API key required")
        
        dataset_id = f"{workspace}/{project}/{version}"
        dataset_path = self.datasets_dir / f"{project}_v{version}"
        
        # Check if already downloaded
        if dataset_path.exists():
            logger.info(f"Dataset {dataset_id} already exists")
            return await self._load_dataset_info(dataset_path)
        
        # Download from Roboflow
        logger.info(f"Downloading dataset: {dataset_id}")
        project_obj = self.rf.workspace(workspace).project(project)
        version_obj = project_obj.version(version)
        
        dataset = version_obj.download(format, location=str(dataset_path))
        
        # Parse and validate
        info = await self._load_dataset_info(dataset_path)
        
        logger.info(f"Dataset downloaded: {info.name}")
        return info
    
    async def load_local_dataset(self, yaml_path: Path) -> DatasetInfo:
        """Load a locally stored dataset from its YAML config."""
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Dataset YAML not found: {yaml_path}")
        
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        dataset_dir = yaml_path.parent
        
        # Count images in each split
        splits = {}
        for split_name in ['train', 'val', 'test']:
            split_path = dataset_dir / split_name / 'images'
            if split_path.exists():
                image_count = len(list(split_path.glob('*')))
                splits[split_name] = DatasetSplit(
                    path=str(split_path),
                    image_count=image_count
                )
        
        return DatasetInfo(
            id=yaml_path.stem,
            name=config.get('names', {}).get(0, 'Unknown'),
            path=str(dataset_dir),
            yaml_path=str(yaml_path),
            num_classes=len(config.get('names', {})),
            class_names=list(config.get('names', {}).values()),
            splits=splits,
            format='yolo'
        )
    
    async def _load_dataset_info(self, dataset_path: Path) -> DatasetInfo:
        """Load dataset info from directory."""
        yaml_files = list(dataset_path.glob('*.yaml'))
        if not yaml_files:
            raise ValueError(f"No YAML config found in {dataset_path}")
        
        return await self.load_local_dataset(yaml_files[0])
    
    def list_datasets(self) -> List[DatasetInfo]:
        """List all locally available datasets."""
        datasets = []
        for yaml_file in self.datasets_dir.rglob('*.yaml'):
            try:
                info = asyncio.run(self.load_local_dataset(yaml_file))
                datasets.append(info)
            except Exception as e:
                logger.warning(f"Failed to load dataset {yaml_file}: {e}")
        
        return datasets
    
    async def delete_dataset(self, dataset_id: str) -> bool:
        """Delete a dataset from local storage."""
        dataset_path = self.datasets_dir / dataset_id
        if dataset_path.exists():
            shutil.rmtree(dataset_path)
            logger.info(f"Deleted dataset: {dataset_id}")
            return True
        return False
    
    def validate_dataset(self, yaml_path: Path) -> Dict[str, any]:
        """Validate dataset structure and return report."""
        report = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        try:
            with open(yaml_path, 'r') as f:
                config = yaml.safe_load(f)
            
            dataset_dir = yaml_path.parent
            
            # Check required keys
            required = ['path', 'train', 'val', 'names']
            for key in required:
                if key not in config:
                    report['errors'].append(f"Missing required key: {key}")
                    report['valid'] = False
            
            # Validate splits
            for split in ['train', 'val', 'test']:
                if split in config:
                    split_path = dataset_dir / config[split]
                    if not split_path.exists():
                        report['warnings'].append(f"{split} path does not exist: {split_path}")
                    else:
                        # Count images
                        image_dir = split_path / 'images'
                        label_dir = split_path / 'labels'
                        
                        if image_dir.exists():
                            image_count = len(list(image_dir.glob('*')))
                            report['stats'][f'{split}_images'] = image_count
                        
                        if label_dir.exists():
                            label_count = len(list(label_dir.glob('*')))
                            report['stats'][f'{split}_labels'] = label_count
            
            # Validate class names
            names = config.get('names', {})
            report['stats']['num_classes'] = len(names)
            
        except Exception as e:
            report['valid'] = False
            report['errors'].append(str(e))
        
        return report
```

### 4.2 Data Preprocessing & Augmentation

```python
# app/data/preprocessor.py
from pathlib import Path
from typing import Tuple, Optional
import cv2
import numpy as np
from PIL import Image
import albumentations as A

class ImagePreprocessor:
    """Preprocess images for training/inference."""
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (640, 640),
        normalize: bool = True,
        enhance_contrast: bool = False
    ):
        self.target_size = target_size
        self.normalize = normalize
        self.enhance_contrast = enhance_contrast
    
    def preprocess(self, image_path: Path) -> np.ndarray:
        """Load and preprocess image."""
        # Load image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize
        image = cv2.resize(image, self.target_size)
        
        # Contrast enhancement for cracks
        if self.enhance_contrast:
            image = self._enhance_crack_contrast(image)
        
        # Normalize
        if self.normalize:
            image = image.astype(np.float32) / 255.0
        
        return image
    
    def _enhance_crack_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance contrast specifically for crack detection."""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        # Convert back to RGB
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return enhanced

class AugmentationPipeline:
    """Data augmentation pipeline for crack detection."""
    
    def __init__(self, augmentation_level: str = 'medium'):
        self.augmentation_level = augmentation_level
        self.transform = self._build_transform()
    
    def _build_transform(self) -> A.Compose:
        """Build augmentation pipeline."""
        
        if self.augmentation_level == 'none':
            return A.Compose([])
        
        base_transforms = [
            # Geometric augmentations
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=15,
                p=0.5
            ),
        ]
        
        if self.augmentation_level == 'medium':
            base_transforms.extend([
                # Photometric augmentations
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                A.GaussNoise(var_limit=(10, 50), p=0.3),
                A.GaussianBlur(blur_limit=3, p=0.3),
                
                # Crack-specific augmentations
                A.RandomShadow(
                    num_shadows_lower=1,
                    num_shadows_upper=2,
                    shadow_dimension=5,
                    p=0.2
                ),
            ])
        
        elif self.augmentation_level == 'heavy':
            base_transforms.extend([
                A.RandomBrightnessContrast(p=0.7),
                A.GaussNoise(var_limit=(10, 100), p=0.5),
                A.GaussianBlur(blur_limit=5, p=0.4),
                A.MotionBlur(blur_limit=5, p=0.3),
                A.RandomShadow(p=0.4),
                A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.2),
                A.RandomRain(p=0.1),
                A.CoarseDropout(
                    max_holes=8,
                    max_height=32,
                    max_width=32,
                    p=0.3
                ),
            ])
        
        return A.Compose(base_transforms)
    
    def apply(self, image: np.ndarray, mask: Optional[np.ndarray] = None):
        """Apply augmentation to image and optional mask."""
        if mask is not None:
            transformed = self.transform(image=image, mask=mask)
            return transformed['image'], transformed['mask']
        else:
            return self.transform(image=image)['image']
```

---

## 5. Training Module

### 5.1 Training Configuration

```python
# app/core/training/config.py
from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from pathlib import Path
from enum import Enum

class ModelType(str, Enum):
    YOLOV8_SEG = "yolov8-seg"
    YOLOV9_SEG = "yolov9-seg"
    YOLOV8N_SEG = "yolov8n-seg"
    YOLOV8S_SEG = "yolov8s-seg"
    YOLOV8M_SEG = "yolov8m-seg"
    YOLOV8L_SEG = "yolov8l-seg"
    YOLOV8X_SEG = "yolov8x-seg"
    RFDETR = "rfdetr"

class OptimizerType(str, Enum):
    SGD = "SGD"
    ADAM = "Adam"
    ADAMW = "AdamW"
    RMSPROP = "RMSprop"

class AugmentationLevel(str, Enum):
    NONE = "none"
    LIGHT = "light"
    MEDIUM = "medium"
    HEAVY = "heavy"

class TrainingConfig(BaseModel):
    """Training configuration schema."""
    
    # Model settings
    model_type: ModelType = ModelType.YOLOV8_SEG
    model_size: str = "n"  # n, s, m, l, x for YOLO
    pretrained: bool = True
    
    # Dataset settings
    dataset_yaml: str
    num_classes: int = 1
    image_size: int = 640
    
    # Training hyperparameters
    epochs: int = Field(default=100, ge=1, le=1000)
    batch_size: int = Field(default=16, ge=1, le=128)
    learning_rate: float = Field(default=0.01, ge=0.0001, le=0.1)
    lr_factor: float = 0.01  # Final LR = lr0 * lrf
    momentum: float = Field(default=0.937, ge=0.0, le=1.0)
    weight_decay: float = Field(default=0.0005, ge=0.0, le=0.1)
    optimizer: OptimizerType = OptimizerType.SGD
    
    # Learning rate schedule
    cosine_lr: bool = True
    warmup_epochs: int = 3
    warmup_momentum: float = 0.8
    
    # Early stopping
    patience: int = 50
    min_delta: float = 0.001
    
    # Augmentation
    augmentation_level: AugmentationLevel = AugmentationLevel.MEDIUM
    mosaic: float = 1.0
    mixup: float = 0.0
    copy_paste: float = 0.0
    close_mosaic: int = 10
    
    # System settings
    device: str = "0"  # GPU device(s)
    num_workers: int = 8
    seed: int = 42
    
    # Checkpointing
    save_period: int = 10
    save_best: bool = True
    
    # Output
    output_dir: str = "./checkpoints"
    project_name: str = "crack-detection"
    
    # Advanced YOLO options
    box_gain: float = 7.5
    cls_gain: float = 0.5
    dfl_gain: float = 1.5
    iou_threshold: float = 0.7
    
    def get_model_path(self) -> str:
        """Get pretrained model path."""
        if self.model_type == ModelType.RFDETR:
            return "rfdetr"
        
        model_map = {
            ModelType.YOLOV8_SEG: f"yolov8{self.model_size}-seg.pt",
            ModelType.YOLOV9_SEG: f"yolov9{self.model_size}-seg.pt",
            ModelType.YOLOV8N_SEG: "yolov8n-seg.pt",
            ModelType.YOLOV8S_SEG: "yolov8s-seg.pt",
            ModelType.YOLOV8M_SEG: "yolov8m-seg.pt",
            ModelType.YOLOV8L_SEG: "yolov8l-seg.pt",
            ModelType.YOLOV8X_SEG: "yolov8x-seg.pt",
        }
        return model_map.get(self.model_type, "yolov8n-seg.pt")
```

### 5.2 Checkpoint Management

```python
# app/core/training/checkpoint.py
import json
import shutil
from pathlib import Path
from typing import List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
import torch

from app.models.database import SessionLocal
from app.models.checkpoint import CheckpointModel

@dataclass
class CheckpointInfo:
    id: str
    session_id: str
    epoch: int
    path: str
    metrics: dict
    is_best: bool
    created_at: datetime
    file_size: int

class CheckpointManager:
    """Manage training checkpoints."""
    
    def __init__(self, checkpoint_dir: Path = Path("./checkpoints")):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(
        self,
        session_id: str,
        epoch: int,
        model_state: dict,
        metrics: dict,
        is_best: bool = False
    ) -> CheckpointInfo:
        """Save a training checkpoint."""
        
        # Create checkpoint directory
        session_dir = self.checkpoint_dir / session_id
        session_dir.mkdir(exist_ok=True)
        
        # Save model weights
        checkpoint_name = f"epoch_{epoch:03d}.pt"
        checkpoint_path = session_dir / checkpoint_name
        
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'metrics': metrics,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        torch.save(checkpoint_data, checkpoint_path)
        
        # Save best model separately
        if is_best:
            best_path = session_dir / "best.pt"
            shutil.copy(checkpoint_path, best_path)
        
        # Save metrics JSON
        metrics_path = session_dir / f"epoch_{epoch:03d}_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Create info
        info = CheckpointInfo(
            id=f"{session_id}_{epoch}",
            session_id=session_id,
            epoch=epoch,
            path=str(checkpoint_path),
            metrics=metrics,
            is_best=is_best,
            created_at=datetime.utcnow(),
            file_size=checkpoint_path.stat().st_size
        )
        
        # Persist to database
        self._save_to_db(info)
        
        return info
    
    def load_checkpoint(self, checkpoint_path: Path) -> dict:
        """Load a checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        return checkpoint
    
    def list_checkpoints(self, session_id: str) -> List[CheckpointInfo]:
        """List all checkpoints for a session."""
        session_dir = self.checkpoint_dir / session_id
        if not session_dir.exists():
            return []
        
        checkpoints = []
        for checkpoint_file in sorted(session_dir.glob("epoch_*.pt")):
            epoch = int(checkpoint_file.stem.split('_')[1])
            metrics_file = session_dir / f"epoch_{epoch:03d}_metrics.json"
            
            metrics = {}
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
            
            info = CheckpointInfo(
                id=f"{session_id}_{epoch}",
                session_id=session_id,
                epoch=epoch,
                path=str(checkpoint_file),
                metrics=metrics,
                is_best=(session_dir / "best.pt").resolve() == checkpoint_file.resolve(),
                created_at=datetime.fromtimestamp(checkpoint_file.stat().st_mtime),
                file_size=checkpoint_file.stat().st_size
            )
            checkpoints.append(info)
        
        return sorted(checkpoints, key=lambda x: x.epoch)
    
    def get_best_checkpoint(self, session_id: str) -> Optional[CheckpointInfo]:
        """Get the best checkpoint for a session."""
        checkpoints = self.list_checkpoints(session_id)
        best = [c for c in checkpoints if c.is_best]
        return best[0] if best else None
    
    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint."""
        # Parse checkpoint_id
        parts = checkpoint_id.rsplit('_', 1)
        if len(parts) != 2:
            return False
        
        session_id, epoch = parts
        session_dir = self.checkpoint_dir / session_id
        
        checkpoint_path = session_dir / f"epoch_{int(epoch):03d}.pt"
        metrics_path = session_dir / f"epoch_{int(epoch):03d}_metrics.json"
        
        deleted = False
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            deleted = True
        if metrics_path.exists():
            metrics_path.unlink()
        
        # Remove from database
        if deleted:
            self._delete_from_db(checkpoint_id)
        
        return deleted
    
    def cleanup_old_checkpoints(self, session_id: str, keep_last_n: int = 5):
        """Remove old checkpoints, keeping only the last N."""
        checkpoints = self.list_checkpoints(session_id)
        
        # Keep best and last N
        to_keep = {c.id for c in checkpoints if c.is_best}
        to_keep.update(c.id for c in checkpoints[-keep_last_n:])
        
        for checkpoint in checkpoints:
            if checkpoint.id not in to_keep:
                self.delete_checkpoint(checkpoint.id)
    
    def _save_to_db(self, info: CheckpointInfo):
        """Save checkpoint info to database."""
        db = SessionLocal()
        try:
            db_checkpoint = CheckpointModel(
                id=info.id,
                session_id=info.session_id,
                epoch=info.epoch,
                path=info.path,
                metrics=info.metrics,
                is_best=info.is_best,
                created_at=info.created_at,
                file_size=info.file_size
            )
            db.merge(db_checkpoint)
            db.commit()
        finally:
            db.close()
    
    def _delete_from_db(self, checkpoint_id: str):
        """Delete checkpoint from database."""
        db = SessionLocal()
        try:
            checkpoint = db.query(CheckpointModel).filter(
                CheckpointModel.id == checkpoint_id
            ).first()
            if checkpoint:
                db.delete(checkpoint)
                db.commit()
        finally:
            db.close()
```

---

## 6. Prediction Module

### 6.1 Inference Pipeline

```python
# app/core/prediction/predictor.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union, Optional
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from dataclasses import dataclass

@dataclass
class DetectionResult:
    """Single detection result."""
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple  # (x1, y1, x2, y2)
    segmentation: Optional[np.ndarray] = None
    area: Optional[float] = None

@dataclass
class PredictionResult:
    """Full prediction result for an image."""
    image_path: str
    image_size: tuple
    detections: List[DetectionResult]
    inference_time: float
    model_name: str
    timestamp: str

class BasePredictor(ABC):
    """Abstract base class for model predictors."""
    
    def __init__(self, model_path: str, device: str = "0"):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.class_names = {}
    
    @abstractmethod
    def load_model(self):
        """Load the model weights."""
        pass
    
    @abstractmethod
    def predict(
        self,
        image: Union[str, Path, np.ndarray],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ) -> PredictionResult:
        """Run inference on an image."""
        pass
    
    @abstractmethod
    def predict_batch(
        self,
        images: List[Union[str, Path, np.ndarray]],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ) -> List[PredictionResult]:
        """Run inference on a batch of images."""
        pass
    
    def visualize(
        self,
        image: np.ndarray,
        result: PredictionResult,
        show_confidence: bool = True,
        show_segmentation: bool = True,
        color_map: Optional[Dict[int, tuple]] = None
    ) -> np.ndarray:
        """Draw detections on image."""
        vis_image = image.copy()
        
        if color_map is None:
            color_map = {0: (0, 255, 0)}  # Default green for cracks
        
        for det in result.detections:
            color = color_map.get(det.class_id, (0, 255, 0))
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, det.bbox)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw segmentation mask
            if show_segmentation and det.segmentation is not None:
                mask = det.segmentation.astype(np.uint8) * 255
                colored_mask = np.zeros_like(vis_image)
                colored_mask[:] = color
                alpha = 0.5
                vis_image = cv2.addWeighted(
                    vis_image, 1,
                    colored_mask, alpha,
                    0, vis_image,
                    mask=mask
                )
            
            # Draw label
            if show_confidence:
                label = f"{det.class_name} {det.confidence:.2f}"
                (text_w, text_h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                cv2.rectangle(
                    vis_image,
                    (x1, y1 - text_h - 10),
                    (x1 + text_w, y1),
                    color, -1
                )
                cv2.putText(
                    vis_image, label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2
                )
        
        return vis_image

# app/core/prediction/yolo_predictor.py
from ultralytics import YOLO
import time
from datetime import datetime

class YOLOPredictor(BasePredictor):
    """YOLO model predictor."""
    
    def load_model(self):
        """Load YOLO model."""
        self.model = YOLO(self.model_path)
        self.class_names = self.model.names
        return self
    
    def predict(
        self,
        image: Union[str, Path, np.ndarray],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ) -> PredictionResult:
        """Run YOLO inference."""
        if self.model is None:
            self.load_model()
        
        start_time = time.time()
        
        results = self.model(
            image,
            conf=conf_threshold,
            iou=iou_threshold,
            device=self.device,
            verbose=False
        )[0]
        
        inference_time = time.time() - start_time
        
        # Parse results
        detections = []
        
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy().astype(int)
            
            # Get segmentation masks if available
            masks = None
            if hasattr(results, 'masks') and results.masks is not None:
                masks = results.masks.data.cpu().numpy()
            
            for i, (box, conf, cls) in enumerate(zip(boxes, confs, classes)):
                det = DetectionResult(
                    class_id=int(cls),
                    class_name=self.class_names.get(int(cls), f"class_{cls}"),
                    confidence=float(conf),
                    bbox=tuple(box),
                    segmentation=masks[i] if masks is not None else None,
                    area=float((box[2] - box[0]) * (box[3] - box[1]))
                )
                detections.append(det)
        
        # Get image info
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            image_path = str(image)
            image_size = (img.shape[1], img.shape[0])
        else:
            image_path = "numpy_array"
            image_size = (image.shape[1], image.shape[0])
        
        return PredictionResult(
            image_path=image_path,
            image_size=image_size,
            detections=detections,
            inference_time=inference_time,
            model_name=self.model_path,
            timestamp=datetime.utcnow().isoformat()
        )
    
    def predict_batch(
        self,
        images: List[Union[str, Path, np.ndarray]],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ) -> List[PredictionResult]:
        """Run batch inference."""
        return [
            self.predict(img, conf_threshold, iou_threshold)
            for img in images
        ]
```

### 6.2 Export Options

```python
# app/core/prediction/export.py
import json
import csv
from pathlib import Path
from typing import List
from PIL import Image
import cv2
import xml.etree.ElementTree as ET

from app.core.prediction.predictor import PredictionResult

class ResultsExporter:
    """Export prediction results in various formats."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_json(
        self,
        results: List[PredictionResult],
        filename: str = "predictions.json"
    ) -> Path:
        """Export results as JSON."""
        output_path = self.output_dir / filename
        
        data = []
        for result in results:
            data.append({
                'image_path': result.image_path,
                'image_size': result.image_size,
                'inference_time': result.inference_time,
                'model_name': result.model_name,
                'timestamp': result.timestamp,
                'detections': [
                    {
                        'class_id': d.class_id,
                        'class_name': d.class_name,
                        'confidence': d.confidence,
                        'bbox': d.bbox,
                        'area': d.area
                    }
                    for d in result.detections
                ]
            })
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        return output_path
    
    def export_csv(
        self,
        results: List[PredictionResult],
        filename: str = "predictions.csv"
    ) -> Path:
        """Export results as CSV."""
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'image_path', 'class_id', 'class_name',
                'confidence', 'x1', 'y1', 'x2', 'y2', 'area'
            ])
            
            for result in results:
                for det in result.detections:
                    writer.writerow([
                        result.image_path,
                        det.class_id,
                        det.class_name,
                        det.confidence,
                        det.bbox[0], det.bbox[1],
                        det.bbox[2], det.bbox[3],
                        det.area
                    ])
        
        return output_path
    
    def export_visualizations(
        self,
        results: List[PredictionResult],
        images: List[Path],
        predictor,
        filename_prefix: str = "vis"
    ) -> List[Path]:
        """Export visualization images."""
        output_paths = []
        
        for result, image_path in zip(results, images):
            img = cv2.imread(str(image_path))
            vis_img = predictor.visualize(img, result)
            
            output_path = self.output_dir / f"{filename_prefix}_{image_path.name}"
            cv2.imwrite(str(output_path), vis_img)
            output_paths.append(output_path)
        
        return output_paths
    
    def export_coco_format(
        self,
        results: List[PredictionResult],
        filename: str = "predictions_coco.json"
    ) -> Path:
        """Export results in COCO format."""
        output_path = self.output_dir / filename
        
        coco_output = {
            'images': [],
            'annotations': [],
            'categories': []
        }
        
        categories = set()
        annotation_id = 1
        
        for img_id, result in enumerate(results):
            # Add image info
            coco_output['images'].append({
                'id': img_id,
                'file_name': Path(result.image_path).name,
                'width': result.image_size[0],
                'height': result.image_size[1]
            })
            
            # Add detections
            for det in result.detections:
                categories.add((det.class_id, det.class_name))
                
                x1, y1, x2, y2 = det.bbox
                width = x2 - x1
                height = y2 - y1
                
                coco_output['annotations'].append({
                    'id': annotation_id,
                    'image_id': img_id,
                    'category_id': det.class_id,
                    'bbox': [x1, y1, width, height],
                    'area': det.area,
                    'score': det.confidence,
                    'iscrowd': 0
                })
                annotation_id += 1
        
        # Add categories
        for cat_id, cat_name in sorted(categories):
            coco_output['categories'].append({
                'id': cat_id,
                'name': cat_name,
                'supercategory': 'none'
            })
        
        with open(output_path, 'w') as f:
            json.dump(coco_output, f, indent=2)
        
        return output_path
```

---

## 7. Dashboard Module

### 7.1 Real-time Metrics Collection

```python
# app/core/monitoring/metrics_collector.py
import asyncio
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import deque
import json

from app.models.database import SessionLocal
from app.models.training_session import TrainingSessionModel

@dataclass
class TrainingMetrics:
    """Training metrics at a point in time."""
    epoch: int
    train_loss: float
    val_loss: float
    train_iou: float
    val_iou: float
    train_map: float
    val_map: float
    learning_rate: float
    epoch_time: float
    timestamp: datetime

@dataclass
class GPUMetrics:
    """GPU metrics at a point in time."""
    gpu_index: int
    utilization: float
    memory_used_gb: float
    memory_total_gb: float
    temperature: float
    power_draw_w: float
    timestamp: datetime

class MetricsCollector:
    """Collect and manage training and system metrics."""
    
    def __init__(
        self,
        max_history: int = 1000,
        gpu_monitor = None
    ):
        self.max_history = max_history
        self.training_metrics: Dict[str, deque] = {}
        self.gpu_metrics: Dict[int, deque] = {}
        self.gpu_monitor = gpu_monitor
        self._subscribers: List[Callable] = []
    
    def subscribe(self, callback: Callable):
        """Subscribe to metrics updates."""
        self._subscribers.append(callback)
    
    def unsubscribe(self, callback: Callable):
        """Unsubscribe from metrics updates."""
        if callback in self._subscribers:
            self._subscribers.remove(callback)
    
    async def _notify_subscribers(self, session_id: str, metrics: TrainingMetrics):
        """Notify all subscribers of new metrics."""
        for callback in self._subscribers:
            try:
                await callback(session_id, metrics)
            except Exception as e:
                print(f"Subscriber error: {e}")
    
    def record_training_metrics(
        self,
        session_id: str,
        metrics: TrainingMetrics
    ):
        """Record training metrics for a session."""
        if session_id not in self.training_metrics:
            self.training_metrics[session_id] = deque(maxlen=self.max_history)
        
        self.training_metrics[session_id].append(metrics)
        
        # Persist to database
        self._persist_training_metrics(session_id, metrics)
        
        # Notify subscribers
        asyncio.create_task(self._notify_subscribers(session_id, metrics))
    
    def record_gpu_metrics(self, gpu_metrics: List[GPUMetrics]):
        """Record GPU metrics."""
        for metric in gpu_metrics:
            if metric.gpu_index not in self.gpu_metrics:
                self.gpu_metrics[metric.gpu_index] = deque(maxlen=self.max_history)
            
            self.gpu_metrics[metric.gpu_index].append(metric)
    
    def get_training_metrics(
        self,
        session_id: str,
        last_n: Optional[int] = None
    ) -> List[TrainingMetrics]:
        """Get training metrics for a session."""
        if session_id not in self.training_metrics:
            return []
        
        metrics = list(self.training_metrics[session_id])
        if last_n:
            metrics = metrics[-last_n:]
        
        return metrics
    
    def get_gpu_metrics(
        self,
        gpu_index: int = 0,
        last_n: Optional[int] = None
    ) -> List[GPUMetrics]:
        """Get GPU metrics history."""
        if gpu_index not in self.gpu_metrics:
            return []
        
        metrics = list(self.gpu_metrics[gpu_index])
        if last_n:
            metrics = metrics[-last_n:]
        
        return metrics
    
    def get_latest_metrics(self, session_id: str) -> Optional[TrainingMetrics]:
        """Get latest metrics for a session."""
        if session_id not in self.training_metrics:
            return None
        
        metrics = self.training_metrics[session_id]
        return metrics[-1] if metrics else None
    
    def get_training_summary(self, session_id: str) -> Dict:
        """Get summary statistics for a training session."""
        metrics = self.get_training_metrics(session_id)
        
        if not metrics:
            return {}
        
        val_maps = [m.val_map for m in metrics if m.val_map is not None]
        val_losses = [m.val_loss for m in metrics if m.val_loss is not None]
        
        return {
            'total_epochs': len(metrics),
            'best_val_map': max(val_maps) if val_maps else None,
            'best_val_map_epoch': metrics[val_maps.index(max(val_maps))].epoch if val_maps else None,
            'final_val_map': val_maps[-1] if val_maps else None,
            'final_val_loss': val_losses[-1] if val_losses else None,
            'total_time': sum(m.epoch_time for m in metrics),
            'avg_epoch_time': sum(m.epoch_time for m in metrics) / len(metrics) if metrics else 0
        }
    
    def _persist_training_metrics(
        self,
        session_id: str,
        metrics: TrainingMetrics
    ):
        """Persist metrics to database."""
        db = SessionLocal()
        try:
            session = db.query(TrainingSessionModel).filter(
                TrainingSessionModel.id == session_id
            ).first()
            
            if session:
                # Update latest metrics
                session.latest_metrics = json.dumps(asdict(metrics))
                session.updated_at = datetime.utcnow()
                db.commit()
        finally:
            db.close()
    
    def clear_session(self, session_id: str):
        """Clear metrics for a session."""
        if session_id in self.training_metrics:
            del self.training_metrics[session_id]
```

### 7.2 Dashboard Service

```python
# app/api/v1/endpoints/dashboard.py
from fastapi import APIRouter, Depends, Query
from typing import List, Optional
from datetime import datetime, timedelta

from app.schemas.dashboard import (
    DashboardSummary,
    TrainingHistory,
    GPUMetricsResponse,
    SystemStatus
)
from app.core.monitoring.metrics_collector import MetricsCollector
from app.core.monitoring.gpu_monitor import GPUMonitor
from app.models.database import get_db

router = APIRouter(prefix="/dashboard", tags=["dashboard"])

@router.get("/summary", response_model=DashboardSummary)
async def get_dashboard_summary(
    db = Depends(get_db),
    gpu_monitor: GPUMonitor = Depends(get_gpu_monitor)
):
    """Get dashboard summary data."""
    
    # Get active training sessions
    active_sessions = db.query(TrainingSessionModel).filter(
        TrainingSessionModel.status == "running"
    ).count()
    
    # Get completed sessions
    completed_sessions = db.query(TrainingSessionModel).filter(
        TrainingSessionModel.status == "completed"
    ).count()
    
    # Get total datasets
    total_datasets = db.query(DatasetModel).count()
    
    # Get available models
    available_models = len(list(Path("./checkpoints").glob("**/best.pt")))
    
    # Get GPU status
    gpu_stats = gpu_monitor.get_current_stats()
    gpu_status = "healthy" if all(
        s.temperature < 85 for s in gpu_stats
    ) else "warning" if all(
        s.temperature < 95 for s in gpu_stats
    ) else "critical"
    
    return DashboardSummary(
        active_sessions=active_sessions,
        completed_sessions=completed_sessions,
        total_datasets=total_datasets,
        available_models=available_models,
        gpu_status=gpu_status,
        gpu_count=len(gpu_stats),
        last_updated=datetime.utcnow()
    )

@router.get("/training-history", response_model=List[TrainingHistory])
async def get_training_history(
    limit: int = Query(10, ge=1, le=100),
    db = Depends(get_db)
):
    """Get recent training session history."""
    
    sessions = db.query(TrainingSessionModel).order_by(
        TrainingSessionModel.created_at.desc()
    ).limit(limit).all()
    
    return [
        TrainingHistory(
            id=session.id,
            model_type=session.model_type,
            dataset_name=session.dataset_name,
            epochs_completed=session.epochs_completed,
            total_epochs=session.total_epochs,
            best_map=session.best_map,
            status=session.status,
            duration_seconds=session.duration_seconds,
            created_at=session.created_at
        )
        for session in sessions
    ]

@router.get("/gpu-metrics", response_model=GPUMetricsResponse)
async def get_gpu_metrics(
    history_minutes: int = Query(10, ge=1, le=60),
    gpu_monitor: GPUMonitor = Depends(get_gpu_monitor)
):
    """Get GPU metrics with history."""
    
    # Get current stats
    current_stats = gpu_monitor.get_current_stats()
    
    # Get historical data
    all_history = gpu_monitor.get_history()
    
    # Filter to requested time window
    cutoff = datetime.utcnow() - timedelta(minutes=history_minutes)
    filtered_history = [
        [asdict(stat) for stat in snapshot if stat.timestamp > cutoff]
        for snapshot in all_history
    ]
    
    return GPUMetricsResponse(
        current=[asdict(stat) for stat in current_stats],
        history=filtered_history,
        update_interval=gpu_monitor.update_interval
    )

@router.get("/system-status", response_model=SystemStatus)
async def get_system_status(
    gpu_monitor: GPUMonitor = Depends(get_gpu_monitor)
):
    """Get overall system status."""
    
    # GPU info
    gpu_summary = gpu_monitor.get_summary()
    gpu_stats = gpu_monitor.get_current_stats()
    
    # Calculate overall utilization
    avg_utilization = sum(s.utilization for s in gpu_stats) / len(gpu_stats) if gpu_stats else 0
    avg_memory = sum(s.memory_used / s.memory_total for s in gpu_stats) / len(gpu_stats) if gpu_stats else 0
    max_temp = max(s.temperature for s in gpu_stats) if gpu_stats else 0
    
    # Determine status
    status = "healthy"
    issues = []
    
    if max_temp > 85:
        status = "warning"
        issues.append(f"High GPU temperature detected: {max_temp}C")
    if max_temp > 95:
        status = "critical"
        issues.append(f"Critical GPU temperature: {max_temp}C")
    
    return SystemStatus(
        status=status,
        issues=issues,
        gpu_count=gpu_summary['device_count'],
        avg_utilization=avg_utilization,
        avg_memory_usage=avg_memory * 100,
        max_temperature=max_temp,
        timestamp=datetime.utcnow()
    )
```

---

## 8. API Design

### 8.1 REST Endpoints

```yaml
# API Specification

## Datasets

### GET /api/v1/datasets
List all available datasets
Response: List[DatasetInfo]

### POST /api/v1/datasets
Import a new dataset from Roboflow
Body: { workspace: str, project: str, version: int, format: str }
Response: DatasetInfo

### GET /api/v1/datasets/{id}
Get dataset details
Response: DatasetInfo

### DELETE /api/v1/datasets/{id}
Delete a dataset
Response: { success: bool }

### POST /api/v1/datasets/{id}/validate
Validate dataset structure
Response: ValidationReport

## Training

### GET /api/v1/training/sessions
List training sessions
Query: ?status=running|completed|failed&limit=10
Response: List[TrainingSession]

### POST /api/v1/training/sessions
Start a new training session
Body: TrainingConfig
Response: { session_id: str, status: str }

### GET /api/v1/training/sessions/{id}
Get training session details
Response: TrainingSession

### DELETE /api/v1/training/sessions/{id}
Stop/cancel a training session
Response: { success: bool }

### GET /api/v1/training/sessions/{id}/metrics
Get training metrics
Query: ?last_n=100
Response: TrainingMetrics

### GET /api/v1/training/sessions/{id}/checkpoints
List checkpoints for session
Response: List[CheckpointInfo]

### POST /api/v1/training/sessions/{id}/checkpoints/{checkpoint_id}/export
Export a checkpoint
Body: { format: "onnx|tensorrt|tflite" }
Response: { download_url: str }

## Prediction

### POST /api/v1/predict
Run prediction on single image
Body: multipart/form-data (image file)
Query: ?model_id=xyz&conf_threshold=0.25
Response: PredictionResult

### POST /api/v1/predict/batch
Run batch prediction
Body: multipart/form-data (multiple images)
Response: List[PredictionResult]

### POST /api/v1/predict/url
Run prediction on image URL
Body: { url: str }
Response: PredictionResult

### POST /api/v1/predict/export
Export prediction results
Body: { results: [], format: "json|csv|coco" }
Response: { download_url: str }

## Models

### GET /api/v1/models
List available trained models
Response: List[ModelInfo]

### GET /api/v1/models/{id}
Get model details
Response: ModelInfo

### DELETE /api/v1/models/{id}
Delete a model
Response: { success: bool }

### POST /api/v1/models/{id}/export
Export model to different format
Body: { format: "onnx|tensorrt|tflite|coreml", optimize: bool }
Response: { download_url: str }

## Dashboard

### GET /api/v1/dashboard/summary
Get dashboard summary
Response: DashboardSummary

### GET /api/v1/dashboard/training-history
Get training history
Query: ?limit=10
Response: List[TrainingHistory]

### GET /api/v1/dashboard/gpu-metrics
Get GPU metrics
Query: ?history_minutes=10
Response: GPUMetricsResponse

### GET /api/v1/dashboard/system-status
Get system status
Response: SystemStatus

## System

### GET /api/v1/system/info
Get system information
Response: SystemInfo

### GET /api/v1/system/gpu
Get GPU information
Response: GPUInfo

### GET /api/v1/system/health
Health check endpoint
Response: { status: "healthy|degraded|unhealthy" }
```

### 8.2 WebSocket Events

```python
# app/api/v1/websockets/router.py
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, Set
import json
import asyncio

router = APIRouter()

class ConnectionManager:
    """Manage WebSocket connections."""
    
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {
            'training': set(),
            'gpu': set(),
            'notifications': set()
        }
    
    async def connect(self, websocket: WebSocket, channel: str):
        await websocket.accept()
        self.active_connections[channel].add(websocket)
    
    def disconnect(self, websocket: WebSocket, channel: str):
        self.active_connections[channel].discard(websocket)
    
    async def broadcast(self, channel: str, message: dict):
        disconnected = []
        for connection in self.active_connections[channel]:
            try:
                await connection.send_json(message)
            except:
                disconnected.append(connection)
        
        # Clean up disconnected clients
        for conn in disconnected:
            self.active_connections[channel].discard(conn)

manager = ConnectionManager()

@router.websocket("/training/{session_id}")
async def training_websocket(websocket: WebSocket, session_id: str):
    """WebSocket for real-time training updates."""
    await manager.connect(websocket, 'training')
    
    try:
        while True:
            # Wait for client messages (ping/keepalive)
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get('action') == 'ping':
                await websocket.send_json({'type': 'pong'})
            elif message.get('action') == 'get_status':
                # Send current training status
                status = await get_training_status(session_id)
                await websocket.send_json({
                    'type': 'status',
                    'data': status
                })
                
    except WebSocketDisconnect:
        manager.disconnect(websocket, 'training')

@router.websocket("/gpu")
async def gpu_websocket(websocket: WebSocket):
    """WebSocket for real-time GPU monitoring."""
    await manager.connect(websocket, 'gpu')
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get('action') == 'subscribe':
                # Client subscribed, will receive broadcasts
                await websocket.send_json({
                    'type': 'subscribed',
                    'channel': 'gpu'
                })
                
    except WebSocketDisconnect:
        manager.disconnect(websocket, 'gpu')

@router.websocket("/notifications")
async def notifications_websocket(websocket: WebSocket):
    """WebSocket for system notifications."""
    await manager.connect(websocket, 'notifications')
    
    try:
        while True:
            await asyncio.sleep(1)  # Keep connection alive
            
    except WebSocketDisconnect:
        manager.disconnect(websocket, 'notifications')

# Broadcasting functions
gpu_monitor = None  # Set during startup
metrics_collector = None  # Set during startup

async def broadcast_gpu_stats(stats: list):
    """Broadcast GPU stats to all connected clients."""
    await manager.broadcast('gpu', {
        'type': 'gpu_stats',
        'timestamp': datetime.utcnow().isoformat(),
        'data': [asdict(s) for s in stats]
    })

async def broadcast_training_metrics(session_id: str, metrics: TrainingMetrics):
    """Broadcast training metrics."""
    await manager.broadcast('training', {
        'type': 'training_metrics',
        'session_id': session_id,
        'data': asdict(metrics)
    })

async def broadcast_training_complete(session_id: str, result: dict):
    """Broadcast training completion."""
    await manager.broadcast('training', {
        'type': 'training_complete',
        'session_id': session_id,
        'data': result
    })
    
    await manager.broadcast('notifications', {
        'type': 'notification',
        'level': 'success',
        'message': f'Training session {session_id} completed successfully'
    })

async def broadcast_notification(level: str, message: str):
    """Broadcast a notification."""
    await manager.broadcast('notifications', {
        'type': 'notification',
        'level': level,  # info, warning, error, success
        'message': message,
        'timestamp': datetime.utcnow().isoformat()
    })
```

### 8.3 WebSocket Event Types

```typescript
// WebSocket message types

// Client -> Server
interface WSMessage {
  action: 'ping' | 'subscribe' | 'unsubscribe' | 'get_status' | 'get_history';
  channel?: string;
  sessionId?: string;
}

// Server -> Client - GPU Stats
interface GPUStatsMessage {
  type: 'gpu_stats';
  timestamp: string;
  data: GPUStats[];
}

// Server -> Client - Training Metrics
interface TrainingMetricsMessage {
  type: 'training_metrics';
  session_id: string;
  data: {
    epoch: number;
    train_loss: number;
    val_loss: number;
    train_iou: number;
    val_iou: number;
    train_map: number;
    val_map: number;
    learning_rate: number;
    epoch_time: number;
    timestamp: string;
  };
}

// Server -> Client - Training Complete
interface TrainingCompleteMessage {
  type: 'training_complete';
  session_id: string;
  data: {
    best_map: number;
    best_epoch: number;
    total_epochs: number;
    checkpoint_path: string;
  };
}

// Server -> Client - Training Progress
interface TrainingProgressMessage {
  type: 'training_progress';
  session_id: string;
  data: {
    current_epoch: number;
    total_epochs: number;
    progress_percent: number;
    estimated_time_remaining: number;
    current_batch: number;
    total_batches: number;
  };
}

// Server -> Client - Notification
interface NotificationMessage {
  type: 'notification';
  level: 'info' | 'warning' | 'error' | 'success';
  message: string;
  timestamp: string;
}

// Server -> Client - Error
interface ErrorMessage {
  type: 'error';
  message: string;
  code?: string;
}
```

---

## 9. Implementation Phases

### Phase 1: Foundation (Week 1-2)

**Goals:** Set up project structure, basic API, and database

**Tasks:**
1. **Project Setup**
   - Initialize Git repository
   - Create backend/frontend directory structure
   - Set up Docker Compose configuration
   - Configure development environment

2. **Backend Foundation**
   - Set up FastAPI application structure
   - Configure database (SQLite for dev, PostgreSQL for prod)
   - Create database models (SQLAlchemy)
   - Set up Alembic migrations
   - Implement basic CRUD operations

3. **Frontend Foundation**
   - Initialize Vite + React + TypeScript project
   - Configure Tailwind CSS
   - Set up routing (React Router)
   - Create layout components (Sidebar, Header, Main)
   - Set up state management (Zustand)

4. **API Foundation**
   - Implement basic REST endpoints
   - Set up WebSocket infrastructure
   - Configure CORS
   - Add request/response logging

**Deliverables:**
- Running development environment
- Basic API with health check
- Frontend skeleton with navigation

---

### Phase 2: Data Pipeline (Week 3-4)

**Goals:** Implement dataset loading and management

**Tasks:**
1. **Roboflow Integration**
   - Install and configure roboflow-python
   - Implement dataset download functionality
   - Create dataset validation logic
   - Handle YOLO format parsing

2. **Dataset Management API**
   - `GET /datasets` - List datasets
   - `POST /datasets` - Import from Roboflow
   - `GET /datasets/{id}` - Get dataset details
   - `DELETE /datasets/{id}` - Remove dataset
   - `POST /datasets/{id}/validate` - Validate structure

3. **Dataset UI**
   - Dataset list page
   - Import modal with Roboflow URL input
   - Dataset viewer with split statistics
   - Validation status display

4. **Storage Management**
   - Implement dataset storage in filesystem
   - Add storage quota management
   - Create cleanup utilities

**Deliverables:**
- Working dataset import from Roboflow
- Dataset management UI
- Dataset validation pipeline

---

### Phase 3: Training Module - Core (Week 5-6)

**Goals:** Implement basic training functionality with YOLO

**Tasks:**
1. **Training Infrastructure**
   - Install Ultralytics library
   - Implement YOLO trainer class
   - Create training configuration schemas
   - Build training session manager

2. **Training API**
   - `POST /training/sessions` - Start training
   - `GET /training/sessions` - List sessions
   - `GET /training/sessions/{id}` - Get session details
   - `DELETE /training/sessions/{id}` - Stop training
   - `GET /training/sessions/{id}/metrics` - Get metrics

3. **Training UI - Configuration**
   - Model selection (YOLOv8/v9 variants)
   - Dataset selector
   - Hyperparameter form
   - Training options panel

4. **Checkpoint Management**
   - Implement checkpoint saving
   - Best model tracking
   - Checkpoint listing API
   - Checkpoint cleanup logic

**Deliverables:**
- Working YOLO training
- Training configuration UI
- Basic metrics collection

---

### Phase 4: GPU Monitoring (Week 7)

**Goals:** Real-time GPU monitoring and dashboard

**Tasks:**
1. **GPU Monitoring Backend**
   - Install and configure pynvml
   - Implement GPU stats collection
   - Create metrics history storage
   - Add threshold alerting

2. **WebSocket Integration**
   - Set up WebSocket endpoints
   - Implement real-time broadcasting
   - Add connection management
   - Create client reconnection logic

3. **Dashboard Backend**
   - Summary statistics endpoint
   - Training history endpoint
   - GPU metrics endpoint
   - System status endpoint

4. **GPU UI Components**
   - VRAM usage gauge
   - Temperature gauge
   - Utilization bar
   - Historical charts

**Deliverables:**
- Real-time GPU monitoring
- Dashboard with GPU stats
- WebSocket infrastructure

---

### Phase 5: Training Module - Advanced (Week 8-9)

**Goals:** Complete training features with real-time updates

**Tasks:**
1. **Training Monitoring**
   - Integrate training callbacks
   - Real-time metrics streaming
   - Progress tracking
   - ETA calculation

2. **Training UI - Active Monitoring**
   - Real-time loss curves
   - Metrics display cards
   - Progress bar with ETA
   - Live console output

3. **RF-DETR Integration**
   - Install RF-DETR library
   - Implement RF-DETR trainer
   - Add model selection in UI
   - Test and validate

4. **Training History**
   - Training history page
   - Session comparison
   - Export training logs
   - Performance analytics

**Deliverables:**
- Real-time training monitoring
- Both YOLO and RF-DETR support
- Complete training history

---

### Phase 6: Prediction Module (Week 10-11)

**Goals:** Inference pipeline and prediction UI

**Tasks:**
1. **Prediction Backend**
   - Implement YOLO predictor
   - Implement RF-DETR predictor
   - Create prediction service
   - Add batch processing

2. **Prediction API**
   - `POST /predict` - Single image
   - `POST /predict/batch` - Batch images
   - `POST /predict/url` - URL input
   - `GET /predict/formats` - List export formats

3. **Prediction UI**
   - Image uploader (drag & drop)
   - Results viewer
   - Detection overlay visualization
   - Confidence threshold slider

4. **Export Functionality**
   - JSON export
   - CSV export
   - COCO format export
   - Visualized images export

**Deliverables:**
- Working inference pipeline
- Prediction UI
- Multiple export formats

---

### Phase 7: Dashboard & Visualization (Week 12-13)

**Goals:** Complete dashboard with all visualizations

**Tasks:**
1. **Dashboard Backend**
   - Complete all dashboard endpoints
   - Add time-series aggregation
   - Implement caching
   - Add analytics calculations

2. **Dashboard UI Components**
   - Training metrics charts (Recharts)
   - GPU historical graphs
   - Model comparison charts
   - Summary cards

3. **Dashboard Pages**
   - Main dashboard view
   - Training history page
   - GPU monitoring page
   - Model performance page

4. **Real-time Updates**
   - WebSocket integration in dashboard
   - Live metric updates
   - Notification system
   - Alert display

**Deliverables:**
- Complete dashboard
- Real-time visualization
- Training analytics

---

### Phase 8: Polish & Production (Week 14-15)

**Goals:** Production readiness and final polish

**Tasks:**
1. **Model Management**
   - Model registry
   - Model versioning
   - Export to different formats (ONNX, TensorRT)
   - Model comparison

2. **Error Handling**
   - Global error boundaries
   - API error handling
   - User-friendly error messages
   - Recovery mechanisms

3. **Testing**
   - Unit tests (pytest, Vitest)
   - Integration tests
   - E2E tests (Playwright)
   - Performance tests

4. **Documentation**
   - API documentation (OpenAPI/Swagger)
   - User guide
   - Deployment guide
   - README

5. **Production Setup**
   - Docker production build
   - Nginx configuration
   - SSL/TLS setup
   - Monitoring setup (Prometheus/Grafana)

**Deliverables:**
- Production-ready application
- Complete test coverage
- Documentation
- Deployment configuration

---

## Appendix: Key Libraries

### Backend Dependencies

```txt
# requirements.txt

# Web Framework
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-multipart==0.0.6
websockets==12.0

# Database
sqlalchemy==2.0.23
alembic==1.12.1
aiosqlite==0.19.0  # For async SQLite

# Data Validation
pydantic==2.5.0
pydantic-settings==2.1.0
email-validator==2.1.0

# ML/DL
ultralytics==8.0.220
# rfdetr  # Install from GitHub
roboflow==1.1.11
albumentations==1.3.1
opencv-python==4.8.1.78
pillow==10.1.0
numpy==1.24.3
torch==2.1.1
torchvision==0.16.1

# GPU Monitoring
pynvml==11.5.0

# Utilities
pyyaml==6.0.1
python-dotenv==1.0.0
aiofiles==23.2.1
tenacity==8.2.3
httpx==0.25.2

# Background Tasks (optional)
celery==5.3.4
redis==5.0.1

# Monitoring (optional)
prometheus-client==0.19.0

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-cov==4.1.0
httpx==0.25.2

# Development
black==23.11.0
isort==5.12.0
flake8==6.1.0
mypy==1.7.1
```

### Frontend Dependencies

```json
// package.json dependencies
{
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-router-dom": "^6.20.0",
    "zustand": "^4.4.7",
    "@tanstack/react-query": "^5.8.0",
    "axios": "^1.6.2",
    "socket.io-client": "^4.7.2",
    "recharts": "^2.10.3",
    "lucide-react": "^0.294.0",
    "@radix-ui/react-dialog": "^1.0.5",
    "@radix-ui/react-dropdown-menu": "^2.0.6",
    "@radix-ui/react-select": "^2.0.0",
    "@radix-ui/react-slider": "^1.1.2",
    "@radix-ui/react-tabs": "^1.0.4",
    "@radix-ui/react-toast": "^1.1.5",
    "@radix-ui/react-tooltip": "^1.0.7",
    "class-variance-authority": "^0.7.0",
    "clsx": "^2.0.0",
    "tailwind-merge": "^2.1.0",
    "date-fns": "^2.30.0",
    "react-dropzone": "^14.2.3"
  },
  "devDependencies": {
    "@types/react": "^18.2.39",
    "@types/react-dom": "^18.2.17",
    "@vitejs/plugin-react": "^4.2.0",
    "typescript": "^5.3.2",
    "vite": "^5.0.4",
    "tailwindcss": "^3.3.6",
    "postcss": "^8.4.32",
    "autoprefixer": "^10.4.16",
    "eslint": "^8.54.0",
    "@typescript-eslint/eslint-plugin": "^6.13.0",
    "@typescript-eslint/parser": "^6.13.0",
    "eslint-plugin-react-hooks": "^4.6.0",
    "eslint-plugin-react-refresh": "^0.4.4",
    "prettier": "^3.1.0",
    "vitest": "^0.34.6",
    "@testing-library/react": "^14.1.2",
    "@testing-library/jest-dom": "^6.1.5",
    "jsdom": "^23.0.1"
  }
}
```

---

## Summary

This implementation plan provides a comprehensive roadmap for building a production-ready crack detection training client. The architecture is designed to be:

1. **Modular** - Each component can be developed and tested independently
2. **Scalable** - Supports multiple models, datasets, and concurrent training sessions
3. **Real-time** - WebSocket integration provides live updates for training and monitoring
4. **Extensible** - Easy to add new model types and export formats
5. **Production-ready** - Includes monitoring, error handling, and deployment configuration

The 15-week timeline assumes a team of 2-3 developers working full-time. Adjust based on your team size and experience level.

