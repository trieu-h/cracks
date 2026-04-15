**Scripts Usage**

**Backend --> training.py**

**Main functions:**
1. start_training_session
2. stop_training_session
3. train_yolo

1. **start_training_session(config, storage, resume_session_id)**

Purpose: Initializes a new PyTorch thread for training a YOLOv26/v11 model or resumes a previously stopped run. Generates a unique execution ID (UUID) and caches initial flags before securely spinning off a background process away from the main UI thread.

Requirements:
* config: Dictionary containing core hyperparameters (epochs, imgsz, batch, learning rates).
* storage: Fast in-memory state tracking active websockets linked to this operation.
* resume_session_id: (Optional) Direct string ID of the past session to seamlessly resume PyTorch tracking from its last checkpoint epoch.

2. **stop_training_session(session_id, storage)**

Purpose: Safely halts an active PyTorch thread at the nearest batch sequence by engaging a `threading.Event` flag, preventing corruption in the weights and safeguarding memory pools.

Requirements:
* session_id: Explicit UUID identifying the active run.
* storage: State tracker holding the event lock references.

---

**Backend --> video_detection.py**

**Main functions:**
1. extract_frames
2. detect_video_frames
3. create_annotated_video
4. run_video_detection

1. **extract_frames(video_path, output_dir, sample_interval)**

Purpose: Scrapes individual frames linearly from uploaded video files at targeted frame intervals through openCV (`cv2`).

Requirements:
* video_path: Source input media file path.
* output_dir: Working destination where isolated `.jpg` images are pooled.
* sample_interval: Determines frame skipping (e.g., sample=10 fetches every 10th frame) to speed up analysis computations without sacrificing accuracy targets.

2. **detect_video_frames(model_path, frame_paths, model_type, conf, progress_callback)**

Purpose: Invokes the baseline YOLO evaluation function over large stacks of isolated frames recursively while maintaining callback state towards the UI.

Requirements:
* model_path: Sourced weight instance (`.pt` file).
* frame_paths: Iterative array indexing target images generated per `extract_frames`.
* conf: Baseline target confidence (0-1.00) determining segmented box thresholds.

3. **create_annotated_video(frame_results, output_path, original_video_path, fps, sample_interval)**

Purpose: Uses the modified, segmented image results from inference and stitches them back together seamlessly leveraging H.264 codecs native to the browser to ensure the UI can display the resulting predictions accurately natively.

Requirements:
* frame_results: Processed visual array containing polygon cracks over images.
* output_path: Where the mp4 media chunk sits finally.

4. **run_video_detection(model_path, video_path, model_type, conf, sample_interval, storage)**

Purpose: The single encapsulating hub wrapping the frame extractions, mass evaluation processing, and video frame-stitching directly onto a temporary runtime thread. Computes resulting total cracks, time metrics, and structural degradation densities.

---

**Backend --> sync.py**

**Main functions:**
1. discover_offline_runs
2. parse_yolo_results_csv

1. **discover_offline_runs()**

Purpose: Operates independently to recursively scan standard `runs/segment` output locations and `/checkpoints`. Will natively parse and incorporate structurally compliant YOLO weights placed there purely manually (ex. dropping finished external runs straight into folder) dynamically into UI visual metric screens.

2. **parse_yolo_results_csv(csv_path)**

Purpose: Scrapes standard Ultralytics native output tables reading parameters safely ensuring robust compatibility across model versions to compute structural formulas like Precision(M), mAP50, and harmonically calculating F-Scores internally logic-wise. 
