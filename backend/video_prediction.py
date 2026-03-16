"""
Video prediction functions - extract frames, run predictions, compile annotated video.
"""
import uuid
import time
from typing import Dict, List
from pathlib import Path
import cv2
import numpy as np
from prediction import predict_yolo, predict_rfdetr


def extract_frames(video_path: str, output_dir: str, sample_interval: int = 1) -> List[str]:
    """
    Extract frames from video at specified interval.
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save extracted frames
        sample_interval: Process every Nth frame (1 = all frames)
    
    Returns:
        List of paths to extracted frame images
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    frame_paths = []
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process every Nth frame based on sample_interval
        if frame_count % sample_interval == 0:
            frame_path = Path(output_dir) / f"frame_{saved_count:06d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            frame_paths.append(str(frame_path))
            saved_count += 1
        
        frame_count += 1
    
    cap.release()
    return frame_paths, frame_count


def predict_video_frames(
    model_path: str, 
    frame_paths: List[str], 
    model_type: str = 'yolo',
    conf: float = 0.25,
    progress_callback=None
) -> List[Dict]:
    """
    Run prediction on each frame.
    
    Args:
        model_path: Path to trained model
        frame_paths: List of frame image paths
        model_type: 'yolo' or 'rfdetr'
        conf: Confidence threshold
        progress_callback: Optional callback function(frame_idx, total_frames)
    
    Returns:
        List of prediction results for each frame
    """
    results = []
    
    for idx, frame_path in enumerate(frame_paths):
        if model_type == 'yolo':
            result = predict_yolo(model_path, frame_path, conf)
        else:
            result = predict_rfdetr(model_path, frame_path, conf)
        
        results.append({
            'frame_idx': idx,
            'frame_path': frame_path,
            'result': result
        })
        
        if progress_callback:
            progress_callback(idx + 1, len(frame_paths))
    
    return results


def create_annotated_video(
    frame_results: List[Dict],
    output_path: str,
    original_video_path: str = None,
    fps: float = None,
    sample_interval: int = 1
) -> str:
    """
    Create annotated video from processed frames.
    
    Args:
        frame_results: List of frame prediction results
        output_path: Path to save annotated video
        original_video_path: Original video to get FPS and codec info
        fps: Frame rate (if None, extracted from original video)
        sample_interval: Frame sampling interval used during processing
    
    Returns:
        Path to annotated video file
    """
    if not frame_results:
        raise ValueError("No frame results provided")
    
    # Get video properties from first annotated frame or original video
    first_result = frame_results[0]['result']
    annotated_path = first_result.get('annotated_image')
    
    if annotated_path and Path(annotated_path).exists():
        sample_frame = cv2.imread(annotated_path)
        height, width = sample_frame.shape[:2]
    else:
        raise ValueError("Could not determine video dimensions from annotated frames")
    
    # Determine FPS
    if fps is None and original_video_path:
        cap = cv2.VideoCapture(original_video_path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
    
    if fps is None:
        fps = 30.0  # Default FPS
    
    # Adjust FPS for sampling interval (playback at original speed)
    effective_fps = fps / sample_interval
    
    # Create video writer with H.264 codec for browser compatibility
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Try H.264 codec first (best browser support), fallback to MPEG-4
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
    out = cv2.VideoWriter(output_path, fourcc, effective_fps, (width, height))
    
    # If H.264 fails, try mp4v
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, effective_fps, (width, height))
    
    # If mp4v fails, try XVID in AVI container
    if not out.isOpened():
        # Change extension to .avi for XVID
        output_path = str(Path(output_path).with_suffix('.avi'))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, effective_fps, (width, height))
    
    if not out.isOpened():
        raise ValueError(f"Could not create video writer for: {output_path}")
    
    # Write each annotated frame
    for frame_result in frame_results:
        result = frame_result['result']
        annotated_path = result.get('annotated_image')
        
        if annotated_path and Path(annotated_path).exists():
            frame = cv2.imread(annotated_path)
            out.write(frame)
    
    out.release()
    return output_path


def run_video_prediction(
    model_path: str,
    video_path: str,
    model_type: str = 'yolo',
    conf: float = 0.25,
    sample_interval: int = 1,
    storage: Dict = None
) -> str:
    """
    Run full video prediction pipeline.
    
    Args:
        model_path: Path to trained model
        video_path: Path to input video
        model_type: 'yolo' or 'rfdetr'
        conf: Confidence threshold
        sample_interval: Process every Nth frame (1 = all frames)
        storage: Optional storage dict to save results
    
    Returns:
        Prediction ID
    """
    prediction_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    
    # Create working directory
    work_dir = Path('./temp_video') / prediction_id
    frames_dir = work_dir / 'frames'
    
    try:
        # Step 1: Extract frames
        print(f"[Video Prediction {prediction_id}] Extracting frames from: {video_path}")
        frame_paths, total_frames = extract_frames(
            video_path, 
            str(frames_dir), 
            sample_interval
        )
        
        if not frame_paths:
            raise ValueError("No frames extracted from video")
        
        print(f"[Video Prediction {prediction_id}] Extracted {len(frame_paths)} frames from {total_frames} total")
        
        # Step 2: Run predictions on each frame
        print(f"[Video Prediction {prediction_id}] Running predictions with {model_type} model")
        frame_results = predict_video_frames(
            model_path, 
            frame_paths, 
            model_type, 
            conf
        )
        
        # Step 3: Create annotated video
        output_dir = Path('./predictions')
        output_dir.mkdir(exist_ok=True)
        annotated_video_path = output_dir / f"video_pred_{prediction_id}.mp4"
        
        print(f"[Video Prediction {prediction_id}] Creating annotated video")
        create_annotated_video(
            frame_results,
            str(annotated_video_path),
            video_path,
            sample_interval=sample_interval
        )
        
        # Calculate statistics
        total_time = time.time() - start_time
        total_detections = sum(
            r['result'].get('num_detections', 0) 
            for r in frame_results
        )
        
        frames_with_detections = sum(
            1 for r in frame_results 
            if r['result'].get('success') and r['result'].get('num_detections', 0) > 0
        )
        
        result = {
            'success': True,
            'model_type': model_type,
            'model_path': model_path,
            'video_path': video_path,
            'annotated_video': str(annotated_video_path),
            'total_frames': total_frames,
            'processed_frames': len(frame_paths),
            'sample_interval': sample_interval,
            'total_time': total_time,
            'frames_per_second': len(frame_paths) / total_time if total_time > 0 else 0,
            'total_detections': total_detections,
            'frames_with_detections': frames_with_detections,
            'frame_results': [
                {
                    'frame_idx': r['frame_idx'],
                    'num_detections': r['result'].get('num_detections', 0),
                    'detections': r['result'].get('detections', [])[:5]  # Limit stored detections
                }
                for r in frame_results
            ]
        }
        
    except Exception as e:
        import traceback
        print(f"[Video Prediction {prediction_id}] Error: {e}")
        print(traceback.format_exc())
        
        result = {
            'success': False,
            'error': str(e),
            'model_type': model_type,
            'model_path': model_path,
            'video_path': video_path
        }
    
    # Store result
    if storage is not None:
        storage['predictions'][prediction_id] = {
            'id': prediction_id,
            'type': 'video',
            'timestamp': time.time(),
            'result': result
        }
    
    return prediction_id
