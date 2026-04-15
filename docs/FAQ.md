# FAQ - Frequently Asked Questions

1. [How do I import a new dataset?](#how-do-i-import-a-new-dataset)
2. [What format do images and labels need to be in?](#what-format-do-images-and-labels-need-to-be-in)
3. [How do I train the model with more images?](#how-do-i-train-the-model-with-more-images)
4. [How do I resume an interrupted training session?](#how-do-i-resume-an-interrupted-training-session)
5. [How are videos processed and frames extracted?](#how-are-videos-processed-and-frames-extracted)
6. [How do I view predictions in 3D?](#how-do-i-view-predictions-in-3d)
7. [How can I import a model trained outside of this application?](#how-can-i-import-a-model-trained-outside-of-this-application)

---

## How do I import a new dataset?
The application relies on local directory mounting. 
- Ensure your dataset folder is placed on the local machine running the backend.
- Open the **Datasets** tab in the vintage UI framework.
- Enter the absolute or relative path to your dataset folder (e.g., `./datasets/my-cracks/`).
- Click **Import**. The application will automatically parse the `data.yaml` file, count the images, and map the classes directly into the SQLite database.

## What format do images and labels need to be in?
The system exclusively relies on the standardized **Ultralytics YOLO Segmentation** format. 

**Folder Structure Schema:**
```text
dataset_directory/
├── data.yaml
├── train/
│   ├── images/  (e.g., image_1.jpg)
│   └── labels/  (e.g., image_1.txt)
└── val/
    ├── images/
    └── labels/
```

**Label format:**
- Must be a text file (`.txt`) matching the exact name of the image.
- Since we are performing segmentation (not bounding boxes), points mapping the polygon outline MUST be normalized recursively between `0.0` and `1.0`.
- Standard layout: `class_id x1 y1 x2 y2 x3 y3 ...`

> [!NOTE]
> If an image doesn't feature any cracks, you do not need to include a `.txt` label file—the model considers the absence of a label as a decisively negative background.

## How do I train the model with more images?
- Gather your field images and label them using any standard segmentation annotator (saving as YOLO format).
- Create or update the localized dataset folder described above, then import it using the **Datasets** tab.
- Navigate to the **Training** tab explicitly.
- Select `YOLO Segment` as the architecture framework.
- Adjust parameters (epochs, batch size) matching your GPU's threshold limits.
- Click **Start Training** and the websocket metrics charting your loss values will populate real-time!

## How do I resume an interrupted training session?
One of the primary capabilities of the Crack Detection Laboratory is its native resumption logic:
- If your training crashes due to power loss or VRAM overload, restart the backend server.
- The previous session will remain stored securely inside the database.
- Navigate to the **Training** portal, locate your crashed session from the table, and execute the **Resume** action.
- The pipeline will dynamically target the `./checkpoints/<session_id>/weights/last.pt` checkpoint caching where it left off!

## How are videos processed and frames extracted?
You do not need external scripts or applications to evaluate videos. The laboratory possesses a built-in OpenCV pipeline mapping predictions securely:
- Visit the **Detection** tab and select Video upload.
- Choose your model standard and your **Sample Interval** (how actively to skip frames to preserve processing speed).
- The `video_detection.py` hook automatically grabs frames according to your interval, executes inferences against the GPU, dynamically patches the segments together using rigorous `H.264` codec formatting, and streams the finished file directly into your browser seamlessly!

## How do I view predictions in 3D?
- Simply run standard inference using an uploaded image via the **Detection** portal.
- Once the segmentation mask executes in the backend and maps severity zones across the cracks, the resulting image is projected instantly onto the Three.js 3D specimen mesh running natively inside your browser. You can click and drag the mesh physically displaying the predictions directly against an industrial environment!

## How can I import a model trained outside of this application?
Our application runs an automatic sync hook natively named `discover_offline_runs()`.
- If a colleague trained a model dynamically elsewhere, simply grab their `train` output directory containing their `results.csv`, `args.yaml` and `.pt` weights.
- Drop this folder directly into the `./checkpoints` directory inside the backend.
- Visit your application interface and the backend will locate the orphaned files, organically construct an entire UI history instance out of the CSV mappings, and permanently map it!
