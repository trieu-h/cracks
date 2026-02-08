"""
Main FastAPI application - all routes in one file, simple functions.
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os
import json
from pathlib import Path

# Import our simple modules
from storage import get_storage, storage
from gpu_monitor import get_gpu_stats, monitor_gpu
from training import start_training_session, stop_training_session
from prediction import run_prediction, predict_yolo

# Create FastAPI app
app = FastAPI(title="Crack Detection API", version="1.0.0")

# CORS - allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Start GPU monitoring on startup
@app.on_event("startup")
async def startup_event():
    """Start GPU monitoring when server starts."""
    import asyncio
    # Store the event loop for WebSocket broadcasts from training threads
    storage['event_loop'] = asyncio.get_running_loop()
    print(f"Stored event loop: {storage['event_loop']}")
    
    def on_gpu_update(stats):
        # Could broadcast to WebSocket here if needed
        pass
    
    monitor_gpu(on_gpu_update, interval=2.0, storage_list=storage['gpu_history'])

# ===== DATASET ENDPOINTS =====

@app.get("/api/datasets")
def list_datasets():
    """List all registered datasets."""
    return list(storage['datasets'].values())

@app.post("/api/datasets/import")
def import_dataset(data: dict):
    """Import a dataset from local folder path."""
    path = data.get('path', '')
    path = Path(path).expanduser()

    if not path or not path.exists():
        return {'success': False, 'error': 'Path does not exist'}

    # Look for YAML config
    yaml_path = path / 'data.yaml'

    if not yaml_path.exists():
        return {'success': False, 'error': 'No YAML config found'}

    # Parse YAML
    try:
        import yaml
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)

        print("config", config)
        dataset_id = path.name

        # Count images
        train_images = 0
        val_images = 0
        test_images = 0

        train_path = path / 'train' / 'images'
        val_path = path / 'val' / 'images'
        test_path = path / 'test' / 'images'

        print(f"train_path: {train_path}\n val_path: {val_path}\n test_path: {test_path}\n")

        if train_path.exists():
            train_images = len(list(train_path.glob('*')))
        if val_path.exists():
            val_images = len(list(val_path.glob('*')))
        if test_path.exists():
            test_images = len(list(test_path.glob('*')))

        print(f"train_images: {train_images}\n val_images: {val_images}\n test_images: {test_images}\n")

        # Store dataset info
        storage['datasets'][dataset_id] = {
            'id': dataset_id,
            'path': str(path),
            'yaml_path': str(yaml_path),
            'name': config.get('names', {})[0],
            'num_classes': len(config.get('names', [])),
            'class_names': config.get('names', []),
            'train_images': train_images,
            'val_images': val_images,
            'test_images': test_images
        }

        return {'success': True, 'dataset': storage['datasets'][dataset_id]}

    except Exception as e:
        return {'success': False, 'error': str(e)}

@app.get("/api/datasets/{dataset_id}")
def get_dataset(dataset_id: str):
    """Get dataset details."""
    if dataset_id in storage['datasets']:
        return storage['datasets'][dataset_id]
    return {'error': 'Dataset not found'}

@app.delete("/api/datasets/{dataset_id}")
def delete_dataset(dataset_id: str):
    """Delete a dataset from storage."""
    if dataset_id in storage['datasets']:
        del storage['datasets'][dataset_id]
        return {'success': True}
    return {'success': False, 'error': 'Dataset not found'}

# ===== TRAINING ENDPOINTS =====

@app.post("/api/training/start")
def start_training(config: dict):
    """Start a training session."""
    try:
        session_id = start_training_session(config, storage)
        return {'success': True, 'session_id': session_id}
    except Exception as e:
        return {'success': False, 'error': str(e)}

@app.post("/api/training/{session_id}/stop")
def stop_training(session_id: str):
    """Stop a training session."""
    success = stop_training_session(session_id, storage)
    return {'success': success}

@app.get("/api/training/{session_id}/status")
def get_training_status(session_id: str):
    """Get training session status."""
    if session_id in storage['training_sessions']:
        session = storage['training_sessions'][session_id]
        return {
            'id': session_id,
            'status': session['status'],
            'current_epoch': session['current_epoch'],
            'total_epochs': session['total_epochs'],
            'config': session['config']
        }
    return {'error': 'Session not found'}

@app.get("/api/training/{session_id}/metrics")
def get_training_metrics(session_id: str):
    """Get training metrics."""
    if session_id in storage['training_sessions']:
        return storage['training_sessions'][session_id]['metrics']
    return []

@app.get("/api/training/sessions")
def list_training_sessions():
    """List all training sessions."""
    sessions = []
    for sid, session in storage['training_sessions'].items():
        sessions.append({
            'id': sid,
            'status': session['status'],
            'current_epoch': session['current_epoch'],
            'total_epochs': session['total_epochs'],
            'model_type': session['config'].get('model_type', 'yolo')
        })
    return sessions

# ===== PREDICTION ENDPOINTS =====

@app.post("/api/prediction")
def create_prediction(data: dict):
    """Run prediction on an image."""
    model_path = data.get('model_path', '')
    image_path = data.get('image_path', '')
    model_type = data.get('model_type', 'yolo')
    conf = data.get('conf', 0.25)

    if not model_path or not image_path:
        return {'success': False, 'error': 'Missing model_path or image_path'}

    prediction_id = run_prediction(model_path, image_path, model_type, conf, storage)

    # Get result
    result = storage['predictions'].get(prediction_id, {})

    return {
        'success': True,
        'prediction_id': prediction_id,
        'result': result.get('result', {})
    }

@app.get("/api/prediction/{prediction_id}")
def get_prediction(prediction_id: str):
    """Get prediction result."""
    if prediction_id in storage['predictions']:
        return storage['predictions'][prediction_id]
    return {'error': 'Prediction not found'}

# ===== MODEL ENDPOINTS =====

@app.get("/api/models")
def list_models():
    """List trained models."""
    models = []
    checkpoints_dir = Path('./checkpoints')

    if checkpoints_dir.exists():
        for session_dir in checkpoints_dir.iterdir():
            if session_dir.is_dir():
                best_path = session_dir / 'weights' / 'best.pt'
                if best_path.exists():
                    models.append({
                        'id': session_dir.name,
                        'path': str(best_path),
                        'name': f"Model {session_dir.name}",
                        'created': session_dir.stat().st_mtime
                    })

    return models

@app.get("/api/models/{model_id}")
def get_model(model_id: str):
    """Get model details."""
    checkpoints_dir = Path('./checkpoints')
    model_dir = checkpoints_dir / model_id

    if model_dir.exists():
        best_path = model_dir / 'weights' / 'best.pt'
        if best_path.exists():
            return {
                'id': model_id,
                'path': str(best_path),
                'size': best_path.stat().st_size
            }

    return {'error': 'Model not found'}

# ===== SYSTEM ENDPOINTS =====

@app.get("/api/system/gpu")
def get_gpu():
    """Get current GPU stats."""
    return get_gpu_stats()

@app.get("/api/system/health")
def health_check():
    """Health check endpoint."""
    return {'status': 'ok'}

# ===== WEBSOCKET ENDPOINTS =====

@app.websocket("/ws/training/{session_id}")
async def training_websocket(websocket: WebSocket, session_id: str):
    print(f"WebSocket connection requested for session: {session_id}")
    """WebSocket for live training updates."""
    await websocket.accept()
    print(f"WebSocket accepted for session: {session_id}")

    # Register WebSocket for this session
    if session_id not in storage['active_websockets']:
        storage['active_websockets'][session_id] = []
    storage['active_websockets'][session_id].append(websocket)
    print(f"WebSocket registered for session {session_id}. Total websockets: {len(storage['active_websockets'][session_id])}")
    print(f"Active sessions: {list(storage['active_websockets'].keys())}")

    try:
        while True:
            # Keep connection alive and handle any client messages
            data = await websocket.receive_text()
            print(f"Received message from client for session {session_id}: {data}")
            # Could handle client commands here
    except WebSocketDisconnect:
        print(f"WebSocket disconnected for session: {session_id}")
        # Remove WebSocket when disconnected
        if session_id in storage['active_websockets']:
            storage['active_websockets'][session_id].remove(websocket)
            print(f"WebSocket removed for session {session_id}")

@app.websocket("/ws/system")
async def system_websocket(websocket: WebSocket):
    """WebSocket for system/GPU updates."""
    await websocket.accept()

    try:
        while True:
            # Send GPU stats every 2 seconds
            stats = get_gpu_stats()
            await websocket.send_json({
                'type': 'gpu_stats',
                'data': stats
            })
            import asyncio
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
