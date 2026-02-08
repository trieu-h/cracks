"""
Prediction functions - YOLO and RF-DETR, no classes.
"""
import uuid
import time
from typing import Dict, List
from pathlib import Path

def predict_yolo(model_path: str, image_path: str, conf: float = 0.25) -> Dict:
    """Simple YOLO prediction with annotated image."""
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
        
        # Run prediction
        start_time = time.time()
        results = model(image_path, conf=conf)
        inference_time = time.time() - start_time
        
        # Extract results
        result = results[0]
        
        detections = []
        if result.boxes is not None:
            for i, box in enumerate(result.boxes):
                detection = {
                    'class_id': int(box.cls),
                    'class_name': result.names[int(box.cls)],
                    'confidence': float(box.conf),
                    'bbox': box.xyxy.tolist()[0] if len(box.xyxy) > 0 else []
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
            annotated_path = str(annotated_path)
        
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

def predict_rfdetr(model_path: str, image_path: str, conf: float = 0.25) -> Dict:
    """RF-DETR prediction placeholder."""
    # TODO: Implement RF-DETR prediction when library is available
    return {
        'success': False,
        'error': 'RF-DETR prediction not yet implemented',
        'model_type': 'rfdetr',
        'model_path': model_path,
        'image_path': image_path
    }

def run_prediction(model_path: str, image_path: str, model_type: str = 'yolo', 
                   conf: float = 0.25, storage: Dict = None) -> str:
    """Run prediction and return prediction ID."""
    prediction_id = str(uuid.uuid4())[:8]
    
    # Run prediction
    if model_type == 'yolo':
        result = predict_yolo(model_path, image_path, conf)
    else:
        result = predict_rfdetr(model_path, image_path, conf)
    
    # Store result
    if storage is not None:
        storage['predictions'][prediction_id] = {
            'id': prediction_id,
            'timestamp': time.time(),
            'result': result
        }
    
    return prediction_id
