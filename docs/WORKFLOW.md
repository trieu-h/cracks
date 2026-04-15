# Application Workflow

This guide details the standard end-to-end workflow for using the Crack Detection Laboratory, from loading data to running predictions.

## 1. Dataset Import
**Navigate to the `Datasets` page.**
- Ensure your dataset folder is structured in the standard Ultralytics YOLO format (using `train/` and `val/` directories with a `data.yaml` config).
- Enter the path to your dataset directory (e.g., `./datasets/crack_data/`).
- Click **Import**. The application will scan the directory, parse your classes from the configuration file, and save the dataset record in the local SQLite database.

## 2. Launch Training
**Navigate to the `Training` page.**
- Once your dataset is imported, you can begin the training process.
- Select your target dataset from the dropdown list.
- Choose your preferred model architecture (e.g., YOLO Segmentation).
- Adjust your basic hyperparameters, such as the number of epochs and batch size, to match your hardware capabilities.
- Click **Start Training**. The backend server will begin the process, and you can monitor real-time charts tracking the model's loss and convergence.

## 3. Review Models
**Navigate to the `Models` page.**
- After your training run completes, open the Models page to view a history of all successfully trained checkpoints.
- This gallery lists important metrics, such as execution time and overall fitness, allowing you to compare different versions.
- *(Note: If you run a training sequence externally, you can drop the generated folder into the backend's `/checkpoints` directory and click **Sync Local Models** to import it automatically.)*

## 4. Run Detection
**Navigate to the `Detection` page.**
- With a trained model ready, you can now evaluate its performance.
- Open the Detection interface and select your custom checkpoint.
- Upload a target media file (such as a `.jpg` image or `.mp4` video).
- Click **Run Detection**. The system will process the media and generate an output file with highlighted segmentation overlays perfectly mapping the detected faults!
