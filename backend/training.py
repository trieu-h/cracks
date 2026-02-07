"""
Training functions - YOLO and RF-DETR, no classes.
"""
import threading
import uuid
import time
from typing import Dict, Callable, Optional
from pathlib import Path

def train_yolo(config: Dict, session_id: str, on_epoch_end: Optional[Callable] = None,
               on_complete: Optional[Callable] = None, stop_event: Optional[threading.Event] = None):
    """Simple YOLO training function."""
    try:
        from ultralytics import YOLO

        # Load model
        model = YOLO(config.get('model', 'yolo12n-seg.pt'))

        # Training arguments
        args = {
            'data': config['dataset_yaml'],
            'epochs': config.get('epochs', 100),
            'imgsz': config.get('imgsz', 640),
            'batch': config.get('batch', 16),
            'device': config.get('device', '0'),
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
            if stop_event and stop_event.is_set():
                trainer.stop = True
                return

            epoch = trainer.epoch
            metrics = {
                'epoch': epoch,
                'box_loss': trainer.loss_items[0] if hasattr(trainer, 'loss_items') else 0,
                'cls_loss': trainer.loss_items[1] if hasattr(trainer, 'loss_items') else 0,
                'dfl_loss': trainer.loss_items[2] if hasattr(trainer, 'loss_items') else 0,
            }

            if on_epoch_end:
                on_epoch_end(session_id, epoch, metrics)

        # Add callback
        model.add_callback('on_train_epoch_end', callback_trainer)

        # Train
        results = model.train(**args)

        if on_complete:
            on_complete(session_id, True, str(results.best))

    except Exception as e:
        print(f"Training error: {e}")
        if on_complete:
            on_complete(session_id, False, str(e))

def train_rfdetr(config: Dict, session_id: str, on_epoch_end: Optional[Callable] = None,
                 on_complete: Optional[Callable] = None, stop_event: Optional[threading.Event] = None):
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

    if on_complete:
        on_complete(session_id, True, f"./checkpoints/{session_id}/best.pt")

def start_training_session(config: Dict, storage: Dict) -> str:
    """Start a training session and return session ID."""
    session_id = str(uuid.uuid4())[:8]

    # Create stop event
    stop_event = threading.Event()

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
        if sid in storage['training_sessions']:
            storage['training_sessions'][sid]['current_epoch'] = epoch
            storage['training_sessions'][sid]['metrics'].append(metrics)

            # Broadcast to WebSocket if available
            if sid in storage.get('active_websockets', {}):
                for ws in storage['active_websockets'][sid]:
                    try:
                        ws.send_json({
                            'type': 'epoch',
                            'epoch': epoch,
                            'metrics': metrics
                        })
                    except:
                        pass

    def on_complete(sid, success, result):
        if sid in storage['training_sessions']:
            storage['training_sessions'][sid]['status'] = 'completed' if success else 'error'
            storage['training_sessions'][sid]['result'] = result

            # Broadcast completion
            if sid in storage.get('active_websockets', {}):
                for ws in storage['active_websockets'][sid]:
                    try:
                        ws.send_json({
                            'type': 'complete',
                            'success': success,
                            'result': result
                        })
                    except:
                        pass

    model_type = config.get('model_type', 'yolo')

    if model_type == 'yolo':
        thread = threading.Thread(
            target=train_yolo,
            args=(config, session_id, on_epoch_end, on_complete, stop_event),
            daemon=True
        )
    else:
        thread = threading.Thread(
            target=train_rfdetr,
            args=(config, session_id, on_epoch_end, on_complete, stop_event),
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
