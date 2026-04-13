"""
Detection functions - YOLO and RF-DETR, no classes.
"""
import uuid
import time
from typing import Dict, List
from pathlib import Path

def detect_yolo(model_path: str, image_path: str, conf: float = 0.25) -> Dict:
    """Simple YOLO detection with annotated image."""
    try:
        from ultralytics import YOLO
        import cv2
        import os
        import numpy as np
        
        print(f"Predicting with model: {model_path}")
        print(f"Image path: {image_path}")
        print(f"Model exists: {os.path.exists(model_path)}")
        print(f"Image exists: {os.path.exists(image_path)}")
        
        # Load model
        model = YOLO(model_path)
        
        # Move model to GPU if available
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        print(f"Using device: {device} for YOLO detection")
        
        # Run detection
        start_time = time.time()
        results = model(image_path, conf=conf)
        inference_time = time.time() - start_time
        
        # Extract results
        result = results[0]
        
        detections = []
        if result.boxes is not None:
            for i, box in enumerate(result.boxes):
                bbox_list = box.xyxy.tolist()[0] if len(box.xyxy) > 0 else []
                area = 0.0
                if len(bbox_list) == 4:
                    width = bbox_list[2] - bbox_list[0]
                    height = bbox_list[3] - bbox_list[1]
                    area = width * height

                severity = "Low"
                if area > 50000:
                    severity = "High"
                elif area > 10000:
                    severity = "Medium"

                detection = {
                    'class_id': int(box.cls),
                    'class_name': result.names[int(box.cls)],
                    'confidence': float(box.conf),
                    'bbox': bbox_list,
                    'area': area,
                    'severity': severity
                }
                
                # Add segmentation mask if available
                if result.masks is not None and i < len(result.masks):
                    detection['mask'] = result.masks.xy[i].tolist() if len(result.masks.xy) > i else None
                
                detections.append(detection)
        
        # Generate annotated image
        annotated_path = None
        if hasattr(result, 'plot'):
            # Use YOLO's built-in plot function
            annotated_img = result.plot()
            
            # Save annotated image
            output_dir = Path('./predictions')
            output_dir.mkdir(exist_ok=True)
            annotated_path = output_dir / f"pred_{uuid.uuid4().hex[:8]}.jpg"
            cv2.imwrite(str(annotated_path), annotated_img)
            annotated_path = annotated_path.as_posix()  # Use forward slashes for Windows compatibility
        
        return {
            'success': True,
            'inference_time': inference_time,
            'model_type': 'yolo',
            'model_path': model_path,
            'image_path': image_path,
            'annotated_image': annotated_path,
            'num_detections': len(detections),
            'detections': detections
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'model_type': 'yolo',
            'model_path': model_path,
            'image_path': image_path
        }

def run_detection(model_path: str, image_path: str, model_type: str = 'yolov26', 
                   conf: float = 0.25, storage: Dict = None) -> str:
    """Run detection and return detection ID."""
    detection_id = str(uuid.uuid4())[:8]
    
    # Run detection exclusively with YOLO logic
    result = detect_yolo(model_path, image_path, conf)
    
    # Store result
    if storage is not None:
        storage['detections'][detection_id] = {
            'id': detection_id,
            'timestamp': time.time(),
            'result': result
        }
    
    return detection_id
