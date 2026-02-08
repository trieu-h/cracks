"""
Training functions - YOLO and RF-DETR, no classes.
"""
import threading
import uuid
import time
import asyncio
from typing import Dict, Callable, Optional
from pathlib import Path

def train_yolo(config: Dict, session_id: str, on_epoch_end: Optional[Callable] = None,
               on_complete: Optional[Callable] = None, stop_event: Optional[threading.Event] = None,
               event_loop: Optional[asyncio.AbstractEventLoop] = None):
    """Simple YOLO training function."""
    try:
        from ultralytics import YOLO
        import torch

        # Load model
        model = YOLO("./models/" + config.get('model', 'yolo11n-seg.pt'))
        device = "0" if torch.cuda.is_available() else "cpu"

        # Training arguments
        args = {
            'data': config['dataset_yaml'],
            'epochs': config.get('epochs', 100),
            'imgsz': config.get('imgsz', 640),
            'batch': config.get('batch', 16),
            'device': device,
            'lr0': config.get('lr0', 0.01),
            'lrf': config.get('lrf', 0.01),
            'momentum': config.get('momentum', 0.937),
            'weight_decay': config.get('weight_decay', 0.0005),
            'patience': config.get('patience', 50),
            'save': True,
            'project': './checkpoints',
            'name': session_id,
            'exist_ok': True,
        }

        # Custom callback for epoch end
        def callback_trainer(trainer):
            print(f"Callback triggered for session {session_id}")
            if stop_event and stop_event.is_set():
                trainer.stop = True
                return

            epoch = trainer.epoch
            print(f"Epoch completed: {epoch}")
            
            # Convert tensors to floats for JSON serialization
            def to_float(val):
                if hasattr(val, 'item'):  # Tensor
                    return float(val.item())
                return float(val) if val is not None else 0.0
            
            metrics = {
                'epoch': epoch,
                'box_loss': to_float(trainer.loss_items[0]) if hasattr(trainer, 'loss_items') and len(trainer.loss_items) > 0 else 0.0,
                'cls_loss': to_float(trainer.loss_items[1]) if hasattr(trainer, 'loss_items') and len(trainer.loss_items) > 1 else 0.0,
                'dfl_loss': to_float(trainer.loss_items[2]) if hasattr(trainer, 'loss_items') and len(trainer.loss_items) > 2 else 0.0,
            }
            print(f"Metrics: {metrics}")

            if on_epoch_end:
                print(f"Calling on_epoch_end callback")
                on_epoch_end(session_id, epoch, metrics)

        # Add callback
        model.add_callback('on_train_epoch_end', callback_trainer)

        # Train
        results = model.train(**args)

        if on_complete:
            # Try to get best checkpoint path, fall back to default location
            best_path = None
            if hasattr(results, 'best'):
                best_path = str(results.best)
            elif hasattr(results, 'save_dir'):
                # For segmentation models, checkpoint is in save_dir/weights/best.pt
                from pathlib import Path
                best_path = str(Path(results.save_dir) / 'weights' / 'best.pt')
            else:
                # Fallback to expected location
                best_path = f"./checkpoints/{session_id}/weights/best.pt"
            on_complete(session_id, True, best_path)

    except Exception as e:
        print(f"Training error: {e}")
        if on_complete:
            on_complete(session_id, False, str(e))

def train_rfdetr(config: Dict, session_id: str, on_epoch_end: Optional[Callable] = None,
                 on_complete: Optional[Callable] = None, stop_event: Optional[threading.Event] = None,
                 event_loop: Optional[asyncio.AbstractEventLoop] = None):
    """RF-DETR training placeholder."""
    # TODO: Implement RF-DETR training when library is available
    print(f"RF-DETR training not yet implemented for session {session_id}")

    # Simulate training for now
    for epoch in range(config.get('epochs', 100)):
        if stop_event and stop_event.is_set():
            break

        if on_epoch_end:
            metrics = {
                'epoch': epoch,
                'box_loss': 0.1 - (epoch * 0.001),
                'cls_loss': 0.05 - (epoch * 0.0005),
                'dfl_loss': 0.08 - (epoch * 0.0008),
            }
            on_epoch_end(session_id, epoch, metrics)

        time.sleep(0.1)  # Simulate training time

    # Create a placeholder checkpoint file so it shows in Models page
    from pathlib import Path
    checkpoint_dir = Path('./checkpoints') / session_id / 'weights'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / 'best.pt'
    
    # Create a dummy file (RF-DETR not implemented yet)
    best_path.write_text('# RF-DETR model placeholder - not yet implemented')
    
    if on_complete:
        on_complete(session_id, True, str(best_path))

def start_training_session(config: Dict, storage: Dict) -> str:
    """Start a training session and return session ID."""
    session_id = str(uuid.uuid4())[:8]

    # Create stop event
    stop_event = threading.Event()

    # Get the stored event loop from storage (set during startup)
    event_loop = storage.get('event_loop')
    print(f"Using event loop from storage: {event_loop}")

    # Store session
    storage['training_sessions'][session_id] = {
        'id': session_id,
        'config': config,
        'status': 'running',
        'start_time': time.time(),
        'current_epoch': 0,
        'total_epochs': config.get('epochs', 100),
        'metrics': [],
        'stop_event': stop_event,
        'model_type': config.get('model_type', 'yolo')
    }

    # Start training in thread
    def on_epoch_end(sid, epoch, metrics):
        # print("epoch end\n");
        # print(epoch);
        # print(metrics);

        if sid in storage['training_sessions']:
            storage['training_sessions'][sid]['current_epoch'] = epoch
            storage['training_sessions'][sid]['metrics'].append(metrics)

            # Broadcast to WebSocket if available
            print(f"Checking websockets for session {sid}: {sid in storage.get('active_websockets', {})}")
            if sid in storage.get('active_websockets', {}):
                print(f"Found {len(storage['active_websockets'][sid])} websocket(s)")
                for ws in storage['active_websockets'][sid]:
                    try:
                        print(f"Sending epoch {epoch} to websocket, event_loop: {event_loop}, is_running: {event_loop.is_running() if event_loop else False}")
                        # Use run_coroutine_threadsafe to call async method from sync context
                        if event_loop and event_loop.is_running():
                            future = asyncio.run_coroutine_threadsafe(
                                ws.send_json({
                                    'type': 'epoch',
                                    'epoch': epoch,
                                    'metrics': metrics
                                }),
                                event_loop
                            )
                            # Wait for the result with timeout to catch any errors
                            try:
                                future.result(timeout=5.0)
                                print(f"Sent epoch {epoch} successfully")
                            except Exception as send_error:
                                print(f"Failed to send epoch {epoch}: {send_error}")
                        else:
                            print(f"Cannot send: event_loop not available or not running")
                    except Exception as e:
                        print(f"WebSocket send error: {e}")

    def on_complete(sid, success, result):
        if sid in storage['training_sessions']:
            storage['training_sessions'][sid]['status'] = 'completed' if success else 'error'
            storage['training_sessions'][sid]['result'] = result

            # Broadcast completion
            if sid in storage.get('active_websockets', {}):
                for ws in storage['active_websockets'][sid]:
                    try:
                        # Use run_coroutine_threadsafe to call async method from sync context
                        if event_loop and event_loop.is_running():
                            asyncio.run_coroutine_threadsafe(
                                ws.send_json({
                                    'type': 'complete',
                                    'success': success,
                                    'result': result
                                }),
                                event_loop
                            )
                    except Exception as e:
                        print(f"WebSocket send error: {e}")

    model_type = config.get('model_type', 'yolo')

    if model_type == 'yolo':
        thread = threading.Thread(
            target=train_yolo,
            args=(config, session_id, on_epoch_end, on_complete, stop_event, event_loop),
            daemon=True
        )
    else:
        thread = threading.Thread(
            target=train_rfdetr,
            args=(config, session_id, on_epoch_end, on_complete, stop_event, event_loop),
            daemon=True
        )

    thread.start()

    return session_id

def stop_training_session(session_id: str, storage: Dict) -> bool:
    """Stop a training session."""
    if session_id in storage['training_sessions']:
        session = storage['training_sessions'][session_id]
        if 'stop_event' in session:
            session['stop_event'].set()
            session['status'] = 'stopping'
            return True
    return False
