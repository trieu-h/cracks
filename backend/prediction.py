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
    """RF-DETR prediction implementation for segmentation only."""
    try:
        from rfdetr import RFDETRSegNano, RFDETRSegSmall, RFDETRSegMedium, RFDETRSegLarge, RFDETRSegXLarge, RFDETRSeg2XLarge
        from PIL import Image
        import numpy as np
        import cv2
        import os
        import torch
        from pathlib import Path
        
        # Force CPU on macOS to avoid MPS "Unsupported Border padding mode" error
        if torch.backends.mps.is_available():
            print("MPS detected but forcing CPU to avoid 'Unsupported Border padding mode' error")
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
            torch.set_default_device('cpu')
        
        print(f"Predicting with RF-DETR segmentation model: {model_path}")
        print(f"Image path: {image_path}")
        print(f"Model exists: {os.path.exists(model_path)}")
        print(f"Image exists: {os.path.exists(image_path)}")
        
        # Load segmentation model - try Medium as default
        if os.path.exists(model_path):
            try:
                print("Loading segmentation model with weights")
                model = RFDETRSegMedium(pretrain_weights=model_path)
            except Exception as load_error:
                print(f"Error loading with weights, using base model: {load_error}")
                model = RFDETRSegMedium()
        else:
            # If no checkpoint exists, use base model
            print("No checkpoint found, using pretrained base model")
            model = RFDETRSegMedium()
        
        # Load and predict
        image = Image.open(image_path)
        start_time = time.time()
        detections = model.predict(image, threshold=conf)
        inference_time = time.time() - start_time
        
        # Convert detections to our format
        detection_list = []
        
        # RF-DETR returns detections in supervision format
        if hasattr(detections, 'xyxy'):
            boxes = detections.xyxy
            class_ids = detections.class_id if hasattr(detections, 'class_id') else []
            confidences = detections.confidence if hasattr(detections, 'confidence') else []
            
            # Get segmentation masks
            masks = None
            if hasattr(detections, 'mask') and detections.mask is not None:
                masks = detections.mask
                print(f"Found segmentation masks: {len(masks)} objects")
            
            for i in range(len(boxes)):
                detection = {
                    'class_id': int(class_ids[i]) if i < len(class_ids) else 0,
                    'class_name': f"class_{int(class_ids[i])}" if i < len(class_ids) else "unknown",
                    'confidence': float(confidences[i]) if i < len(confidences) else 0.5,
                    'bbox': boxes[i].tolist() if hasattr(boxes[i], 'tolist') else list(boxes[i])
                }
                
                # Always add mask for segmentation
                if masks is not None and i < len(masks):
                    mask = masks[i]
                    detection['mask'] = mask.tolist() if hasattr(mask, 'tolist') else mask
                
                detection_list.append(detection)
        
        # Generate annotated image using OpenCV
        annotated_path = None
        try:
            img_array = np.array(image)
            if len(img_array.shape) == 2:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
            elif img_array.shape[2] == 4:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
            elif img_array.shape[2] == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # Draw segmentation masks
            if masks is not None:
                print("Drawing segmentation masks")
                overlay = img_array.copy()
                colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
                
                for idx, det in enumerate(detection_list):
                    if 'mask' in det:
                        mask = np.array(det['mask'])
                        if mask.ndim == 2:
                            color = colors[idx % len(colors)]
                            # Create colored mask overlay
                            mask_bool = mask > 0.5 if mask.dtype == np.float32 else mask > 0
                            overlay[mask_bool] = color
                
                # Blend overlay with original
                cv2.addWeighted(overlay, 0.4, img_array, 0.6, 0, img_array)
            
            # Draw bounding boxes and labels
            for det in detection_list:
                bbox = det['bbox']
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{det['class_name']}: {det['confidence']:.2f}"
                cv2.putText(img_array, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Save annotated image
            output_dir = Path('./predictions')
            output_dir.mkdir(exist_ok=True)
            annotated_path = output_dir / f"pred_{uuid.uuid4().hex[:8]}.jpg"
            cv2.imwrite(str(annotated_path), img_array)
            annotated_path = str(annotated_path)
        except Exception as annot_error:
            print(f"Error creating annotation: {annot_error}")
        
        return {
            'success': True,
            'inference_time': inference_time,
            'model_type': 'rfdetr',
            'task': 'segmentation',
            'model_path': model_path,
            'image_path': image_path,
            'annotated_image': annotated_path,
            'num_detections': len(detection_list),
            'detections': detection_list
        }
        
    except ImportError as e:
        return {
            'success': False,
            'error': f'RF-DETR library not installed. Install with: pip install rfdetr. Error: {e}',
            'model_type': 'rfdetr',
            'model_path': model_path,
            'image_path': image_path
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e),
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
