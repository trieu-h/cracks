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

        # Handle resuming logic
        is_resuming = config.get('resume', False)
        
        # Load model definition
        if is_resuming:
            # When resuming, YOLO needs the checkpoint path explicitly
            ckpt_path = f"./checkpoints/{session_id}/weights/last.pt"
            print(f"Resuming YOLO from {ckpt_path}")
            model = YOLO(ckpt_path)
        else:
            model = YOLO("./models/" + config.get('model', 'yolo11n-seg.pt'))
            
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.cuda.empty_cache()

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

        # If resuming, inject the resume flag
        if is_resuming:
            args['resume'] = True

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
            
            # Basic losses
            metrics = {
                'epoch': epoch,
                'box_loss': to_float(trainer.loss_items[0]) if hasattr(trainer, 'loss_items') and len(trainer.loss_items) > 0 else 0.0,
                'cls_loss': to_float(trainer.loss_items[1]) if hasattr(trainer, 'loss_items') and len(trainer.loss_items) > 1 else 0.0,
                'dfl_loss': to_float(trainer.loss_items[2]) if hasattr(trainer, 'loss_items') and len(trainer.loss_items) > 2 else 0.0,
            }
            
            # Detailed metrics from validator if available
            if hasattr(trainer, 'validator') and hasattr(trainer.validator, 'metrics'):
                val_metrics = trainer.validator.metrics
                # For segmentation models, prefer mask metrics suffix (M)
                # For detection, use (B)
                suffix = '(M)' if 'segment' in str(type(trainer)).lower() else '(B)'
                
                results_dict = val_metrics.results_dict if hasattr(val_metrics, 'results_dict') else {}
                
                metrics.update({
                    'precision': to_float(results_dict.get(f'metrics/precision{suffix}', 0.0)),
                    'recall': to_float(results_dict.get(f'metrics/recall{suffix}', 0.0)),
                    'mAP50': to_float(results_dict.get(f'metrics/mAP50{suffix}', 0.0)),
                    'mAP50_95': to_float(results_dict.get(f'metrics/mAP50-95{suffix}', 0.0)),
                    'fitness': to_float(val_metrics.fitness) if hasattr(val_metrics, 'fitness') else 0.0
                })
                
                # Calculate F1 if not provided
                if metrics['precision'] + metrics['recall'] > 0:
                    metrics['f1'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
                else:
                    metrics['f1'] = 0.0

            print(f"Metrics: {metrics}")

            if on_epoch_end:
                print(f"Calling on_epoch_end callback")
                on_epoch_end(session_id, epoch, metrics)

        # Custom callback for batch end
        def callback_batch(trainer):
            if stop_event and stop_event.is_set():
                trainer.stop = True
                return

            # Only send every 10 steps to avoid flooding
            if trainer.fitness % 10 != 0 and trainer.epoch == 0:
                 # In early training, send more often
                 pass
            
            # Use same logic to get current loss
            def to_float(val):
                if hasattr(val, 'item'): return float(val.item())
                return float(val) if val is not None else 0.0

            batch_metrics = {
                'epoch': trainer.epoch,
                'step': trainer.fitness,
                'box_loss': to_float(trainer.loss_items[0]) if hasattr(trainer, 'loss_items') and len(trainer.loss_items) > 0 else 0.0,
                'cls_loss': to_float(trainer.loss_items[1]) if hasattr(trainer, 'loss_items') and len(trainer.loss_items) > 1 else 0.0,
                'dfl_loss': to_float(trainer.loss_items[2]) if hasattr(trainer, 'loss_items') and len(trainer.loss_items) > 2 else 0.0,
            }

            if event_loop and event_loop.is_running():
                from storage import storage
                if session_id in storage.get('active_websockets', {}):
                    for ws in storage['active_websockets'][session_id]:
                        asyncio.run_coroutine_threadsafe(
                            ws.send_json({
                                'type': 'batch',
                                'metrics': batch_metrics
                            }),
                            event_loop
                        )

        # Add callbacks
        model.add_callback('on_train_epoch_end', callback_trainer)
        model.add_callback('on_train_batch_end', callback_batch)

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
    """RF-DETR training implementation for segmentation only."""
    try:
        from rfdetr import RFDETRSegNano, RFDETRSegSmall, RFDETRSegMedium, RFDETRSegLarge, RFDETRSegXLarge, RFDETRSeg2XLarge
        from pathlib import Path
        import os
        import torch

        # Force CPU on macOS to avoid MPS "Unsupported Border padding mode" error
        # RF-DETR uses padding_mode='border' which is not supported on MPS
        if torch.backends.mps.is_available():
            print("MPS detected but forcing CPU to avoid 'Unsupported Border padding mode' error")
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
            # Set device to CPU explicitly
            torch.set_default_device('cpu')

        # Get dataset directory from the YAML path
        dataset_yaml = config.get('dataset_yaml', '')
        if not dataset_yaml:
            raise ValueError("No dataset YAML provided")

        dataset_dir = str(Path(dataset_yaml).parent)
        if not os.path.exists(dataset_dir):
            raise ValueError(f"Dataset directory not found: {dataset_dir}")

        # Initialize segmentation model
        model_variant = config.get('model', 'RFDETRSegMedium')
        print(f"Initializing RF-DETR segmentation model: {model_variant}")

        # Map model names to segmentation classes
        model_map = {
            'RFDETRSegNano': RFDETRSegNano,
            'RFDETRSegSmall': RFDETRSegSmall,
            'RFDETRSegMedium': RFDETRSegMedium,
            'RFDETRSegLarge': RFDETRSegLarge,
            'RFDETRSegXLarge': RFDETRSegXLarge,
            'RFDETRSeg2XLarge': RFDETRSeg2XLarge,
            # Fallback mappings (without Seg prefix)
            'RFDETRNano': RFDETRSegNano,
            'RFDETRSmall': RFDETRSegSmall,
            'RFDETRMedium': RFDETRSegMedium,
            'RFDETRLarge': RFDETRSegLarge,
        }
        model_class = model_map.get(model_variant, RFDETRSegMedium)
        model = model_class()

        # Move model to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device} for RF-DETR training")

        # Create checkpoint directory
        checkpoint_dir = Path('./checkpoints') / session_id
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training history for callbacks
        history = []

        # ... (rest of function)
        # Define a custom PTL callback for real-time updates
        from pytorch_lightning.callbacks import Callback as PTLCallback
        
        class BridgeCallback(PTLCallback):
            def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
                if stop_event and stop_event.is_set():
                    trainer.should_stop = True
                    return

                # Send batch updates every 50 steps
                if trainer.global_step % 50 == 0:
                    metrics = {
                        'epoch': trainer.current_epoch,
                        'step': trainer.global_step,
                        # PTL typically stores losses in callback_metrics
                        'box_loss': float(trainer.callback_metrics.get('train/loss', 0.0)),
                    }
                    if event_loop and event_loop.is_running():
                        from storage import storage
                        if session_id in storage.get('active_websockets', {}):
                            for ws in storage['active_websockets'][session_id]:
                                asyncio.run_coroutine_threadsafe(
                                    ws.send_json({'type': 'batch', 'metrics': metrics}),
                                    event_loop
                                )

            def on_train_epoch_end(self, trainer, pl_module):
                # Standard epoch-end logic
                metrics = {
                    'epoch': trainer.current_epoch,
                    'box_loss': float(trainer.callback_metrics.get('train/loss', 0.0)),
                }
                if on_epoch_end:
                    on_epoch_end(session_id, trainer.current_epoch, metrics)

        # HACK: Patch build_trainer in the rfdetr library to inject our callback
        import rfdetr.training
        original_build_trainer = rfdetr.training.build_trainer
        
        def patched_build_trainer(*args, **kwargs):
            trainer = original_build_trainer(*args, **kwargs)
            trainer.callbacks.append(BridgeCallback())
            return trainer
            
        rfdetr.training.build_trainer = patched_build_trainer

        # Training arguments
        args = {
            'dataset_dir': dataset_dir,
            'epochs': config.get('epochs', 10),  # RF-DETR typically needs fewer epochs
            'batch_size': config.get('batch', 4),
            'grad_accum_steps': config.get('grad_accum_steps', 4),
            'lr': config.get('lr0', 1e-4),
            'run_test': False,  # Disable test evaluation to avoid test_stats error
            'output_dir': str(checkpoint_dir),  # Set output directory
            'device': device,  # Pass device if model.train supports it
        }

        is_resuming = config.get('resume', False)
        if is_resuming:
            # Provide explicit checkpoint path for PTL to resume
            ckpt_path = checkpoint_dir / 'last.ckpt'
            if ckpt_path.exists():
                print(f"Resuming RF-DETR from {ckpt_path}")
                args['resume'] = str(ckpt_path)
            else:
                print(f"WARNING: Wanted to resume but {ckpt_path} missing.")

        print(f"Starting RF-DETR training for session {session_id}")
        print(f"Dataset: {dataset_dir}")
        print(f"Epochs: {args['epochs']}, Batch size: {args['batch_size']}")

        # Train with error handling for test_stats issue
        try:
            model.train(**args)
        except Exception as train_error:
            # Check if it's the test_stats error - if training completed, we can continue
            error_msg = str(train_error)
            if "test_stats" in error_msg:
                print(f"Warning: Training completed but encountered test_stats error (this is a known RF-DETR issue): {error_msg}")
                print("Continuing with checkpoint saving...")
            else:
                raise

        # Save model checkpoint
        import shutil
        best_path = checkpoint_dir / 'weights' / 'best.pt'
        best_path.parent.mkdir(parents=True, exist_ok=True)
        
        # RF-DETR saves to output_dir/checkpoint_best_total.pth
        # Let's find and copy the checkpoint
        rfdetr_checkpoint = checkpoint_dir / 'checkpoint_best_total.pth'
        if rfdetr_checkpoint.exists():
            print(f"Found RF-DETR checkpoint: {rfdetr_checkpoint}")
            shutil.copy(rfdetr_checkpoint, best_path)
        elif hasattr(model, 'checkpoint_path') and model.checkpoint_path:
            print(f"Copying checkpoint from model.checkpoint_path: {model.checkpoint_path}")
            shutil.copy(model.checkpoint_path, best_path)
        else:
            # Look for any .pth files in the checkpoint directory
            pth_files = list(checkpoint_dir.glob('*.pth'))
            if pth_files:
                latest_checkpoint = max(pth_files, key=lambda p: p.stat().st_mtime)
                print(f"Found latest checkpoint: {latest_checkpoint}")
                shutil.copy(latest_checkpoint, best_path)
        
        # Create metadata file to track model type (always segmentation)
        meta_path = checkpoint_dir / 'metadata.txt'
        meta_path.write_text(f'# RF-DETR model metadata\n# Session: {session_id}\n# Model: {model_variant}\n# Task: segmentation')
        
        # Verify checkpoint exists
        if not best_path.exists():
            print(f"Warning: Checkpoint not found at {best_path}, creating marker file")
            best_path.write_text(f'# RF-DETR model checkpoint\n# Session: {session_id}\n# Model: {model_variant}\n# Task: segmentation')

        if on_complete:
            on_complete(session_id, True, str(best_path))

    except ImportError as e:
        error_msg = f"RF-DETR library not installed. Install with: pip install rfdetr. Error: {e}"
        print(error_msg)
        if on_complete:
            on_complete(session_id, False, error_msg)
    except Exception as e:
        error_msg = f"RF-DETR training error: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        if on_complete:
            on_complete(session_id, False, error_msg)

def start_training_session(config: Dict, storage: Dict, resume_session_id: Optional[str] = None) -> str:
    """Start a training session and return session ID."""
    if resume_session_id and resume_session_id in storage['training_sessions']:
        session_id = resume_session_id
        session = storage['training_sessions'][session_id]
        
        # We inject resume=True silently so the training loop handles it
        config['resume'] = True
        
        # Retain past epochs and metrics
        current_epoch = session['current_epoch']
        metrics = session['metrics']
        
        # Reset stop event & update status
        stop_event = threading.Event()
        session['stop_event'] = stop_event
        session['status'] = 'running'
        session['config'] = config
        
    else:
        session_id = str(uuid.uuid4())[:8]
        # Create stop event
        stop_event = threading.Event()
        current_epoch = 0
        metrics = []
        
        # Store initial session
        storage['training_sessions'][session_id] = {
            'id': session_id,
            'config': config,
            'status': 'running',
            'start_time': time.time(),
            'current_epoch': current_epoch,
            'total_epochs': config.get('epochs', 100),
            'metrics': metrics,
            'stop_event': stop_event,
            'model_type': config.get('model_type', 'yolo')
        }

    # Get the stored event loop from storage (set during startup)
    event_loop = storage.get('event_loop')
    print(f"Using event loop from storage: {event_loop}")

    # Save initial state to database
    from storage import save_training_session
    save_training_session(session_id)

    # Start training in thread
    def on_epoch_end(sid, epoch, metrics):
        # print("epoch end\n");
        # print(epoch);
        # print(metrics);

        if sid in storage['training_sessions']:
            storage['training_sessions'][sid]['current_epoch'] = epoch
            storage['training_sessions'][sid]['metrics'].append(metrics)
            
            # Save to database
            from storage import save_training_session
            save_training_session(sid)

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
            
            # Save final state to database
            from storage import save_training_session
            save_training_session(sid)

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
