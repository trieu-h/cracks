import os
import csv
import json
from pathlib import Path
import time
from storage import get_storage, save_training_session

def parse_yolo_results_csv(csv_path):
    """Parse Ultralytics YOLO results.csv into a metrics array."""
    metrics = []
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            # Strip whitespace from header keys
            reader.fieldnames = [name.strip() for name in reader.fieldnames]
            
            for row in reader:
                # Extract basic metrics safely
                epoch = int(float(row.get('epoch', 0)))
                box_loss = float(row.get('train/box_loss', row.get('train/box_loss ', 0.0)))
                cls_loss = float(row.get('train/cls_loss', row.get('train/cls_loss ', 0.0)))
                dfl_loss = float(row.get('train/dfl_loss', row.get('train/dfl_loss ', 0.0)))
                
                prec = float(row.get('metrics/precision(B)', row.get('metrics/precision(M)', 0.0)))
                rec = float(row.get('metrics/recall(B)', row.get('metrics/recall(M)', 0.0)))
                map50 = float(row.get('metrics/mAP50(B)', row.get('metrics/mAP50(M)', 0.0)))
                map50_95 = float(row.get('metrics/mAP50-95(B)', row.get('metrics/mAP50-95(M)', 0.0)))
                fit = float(row.get('fitness', 0.0))
                
                f1 = 0.0
                if (prec + rec) > 0:
                    f1 = 2 * (prec * rec) / (prec + rec)
                    
                metrics.append({
                    'epoch': epoch,
                    'box_loss': box_loss,
                    'cls_loss': cls_loss,
                    'dfl_loss': dfl_loss,
                    'precision': prec,
                    'recall': rec,
                    'mAP50': map50,
                    'mAP50_95': map50_95,
                    'fitness': fit,
                    'f1': f1
                })
    except Exception as e:
        print(f"Failed to parse {csv_path}: {e}")
    return metrics

def discover_offline_runs():
    """Scan directories for external runs and inject them into SQLite."""
    storage = get_storage()
    runs_to_check = []
    
    # Check YOLO standard run directories recursively
    for base in ['./runs/segment', './runs/detect', './checkpoints']:
        base_path = Path(base)
        if base_path.exists():
            for csv_path in base_path.rglob('results.csv'):
                runs_to_check.append(csv_path.parent)

    for run_dir in runs_to_check:
        session_id = run_dir.name
        # Some folders might be named 'train' inside 'segment' or they are actual session IDs
        if session_id in ('train', 'train2', 'val', 'tune'):
            session_id = f"yolo_{session_id}_{int(run_dir.stat().st_mtime)}"
        
        # Skip if already in database
        if session_id in storage['training_sessions']:
            continue
            
        csv_path = run_dir / 'results.csv'
        if csv_path.exists():
            print(f"Discovered offline run: {session_id} at {csv_path}")
            
            metrics = parse_yolo_results_csv(csv_path)
            total_epochs = len(metrics)
            
            # Determine if YOLO or RF-DETR based on path or presence of metadata
            model_type = 'yolo'
            if 'checkpoints' in str(run_dir) and (run_dir / 'metadata.txt').exists():
                model_type = 'rfdetr'
                
            storage['training_sessions'][session_id] = {
                'id': session_id,
                'status': 'completed',
                'current_epoch': total_epochs - 1 if total_epochs > 0 else 0,
                'total_epochs': total_epochs,
                'model_type': model_type,
                'start_time': run_dir.stat().st_mtime,
                'config': {'model_type': model_type, 'imported': True, 'path': str(run_dir)},
                'metrics': metrics
            }
            save_training_session(session_id)
            print(f"Successfully synced offline run {session_id} to database.")
