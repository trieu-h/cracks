# FAQ - Frequently Asked Questions

1. [How do I import a new dataset?](#how-do-i-import-a-new-dataset)
2. [How should my dataset folders be structured?](#how-should-my-dataset-folders-be-structured)
3. [How do I train the model with new images?](#how-do-i-train-the-model-with-new-images)
4. [What happens if the training crashes or is stopped?](#what-happens-if-the-training-crashes-or-is-stopped)
5. [Can I test the model on video files?](#can-i-test-the-model-on-video-files)
6. [Can I import a model trained outside of this app?](#can-i-import-a-model-trained-outside-of-this-app)

---

## How do I import a new dataset?
Your dataset must be located on the same physical machine running the backend application.
- Place your dataset folder anywhere on your computer.
- Click on the **Datasets** tab in the UI.
- Enter the location path to your folder (for example: `./datasets/crack-data/`).
- Click **Import**. The system will index the images and save the configuration to the local database.

## How should my dataset folders be structured?
The system adheres to the standard **YOLO format**. 

**Required Folder Layout:**
```text
dataset_directory/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
└── val/
    ├── images/
    └── labels/
```

**Label Rules:**
- Every image requires a matching text file (`.txt`) in the labels folder.
- If an image has no cracks, do not include a label file. The model will automatically treat it as a negative background image.

## How do I train the model with new images?
- Annotate your new images using any standard tool and export them in the YOLO format.
- Update your dataset directory with the new images, then re-import or verify it on the **Datasets** tab.
- Navigate to the **Training** tab and select the dataset.
- Click **Start Training** to monitor the progress via the live UI charts.

## What happens if the training crashes or is stopped?
The application is designed specifically to prevent data loss.
- If the hardware loses power or exceeds target memory tolerances, safely restart the backend server.
- Your previous session progress remains secure in the database.
- Navigate to the **Training** tab, find the interrupted session, and click **Resume**. The model will load the last saved checkpoint and continue.

## Can I test the model on video files?
Yes, video processing is natively supported.
- Go to the **Detection** tab and select a video to upload.
- Choose a Sample Interval (e.g., skip every 5 frames) to balance processing speed with accuracy.
- The server will extract the frames, run the trained model to detect the cracks, and encode a new annotated `.mp4` video for distribution.

## Can I import a model trained outside of this app?
Yes, using the offline sync feature. 
- Obtain the results folder from the external training session (ensure it contains `results.csv`, `args.yaml` and the `.pt` weight file).
- Move this folder into the `./checkpoints` directory of your backend backend environment.
- Open the application and click **Sync Local Models**. The app will parse the external metrics and map it into your dashboard history.
