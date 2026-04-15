# Standard Application Workflow

This document outlines the standard end-to-end sequence for utilizing the Crack Detection Laboratory efficiently, starting from data ingestion down to target predictions.

## 1. Dataset Import
**Navigate to the `Datasets` page.**
- First, format your cracked images and labels into standard Ultralytics YOLO segmentation directory structures locally.
- Enter the explicit path to your local dataset directory interface (e.g., `./datasets/crack_data/`).
- Click **Import**. The system will scan the folder, fetch the `data.yaml` variables, and safely record the dataset into your internal SQLite database list.

## 2. Launch Training
**Navigate to the `Training` page.**
- Once dataset ingestion confirms successfully, shift to the Training portal.
- Select your target dataset from the dropdown menu (it should appear dynamically).
- Pick one of the two native backbone architectures tailored for segmentation mappings.
- Adjust specific hyper-parameters (such as Epochs or Batch-Size optimizations).
- Click **Start Training** and let the PyTorch execution begin. You can observe the live WebSocket metrics charting your loss patterns immediately!

## 3. Review Checkpoints
**Navigate to the `Models` page.**
- Once the target epochs complete, open the **Models** tab.
- Here you'll witness a history gallery of your fully trained metrics synced naturally for comparison.
- This tab securely points back to the internal `/checkpoints` hub and displays specific UUID metrics linking to your exported `.pt` weight strings.
- *(Note: If you have an external trained folder structure, dropping it into the backend folders and clicking 'Sync Local Models' natively builds the list out as well.)*

## 4. Run Detection
**Navigate to the `Detection` page to test.**
- Now that your custom checkpoints are secured, test them!
- Select the `Detection` testing panel.
- Upload any target media file, such as a localized image (`.jpg`/`.png`) or a drone-inspection video (`.mp4`).
- Optionally scale the confidence thresholds natively to restrict noise filtering.
- Click **Run Detection**. The backend will spin up evaluation threads dynamically wrapping colored crack polygon overlays mapping where the damage originates over the surface of your uploaded image/video!
