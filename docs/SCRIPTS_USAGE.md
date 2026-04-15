# Developer Scripts (Simple Explanations)

This file explains the invisible background scripts that make the app work. You do not need to read this to use the app! This is just for people who want to understand the code.

## 1. The Training Script (`training.py`)

This file is responsible for teaching the AI.

**What it does:**
- `start_training_session`: This is like the "Start" button under the hood. It prepares the AI, collects your settings, and tells your computer's graphics card to start learning. It runs in the background so you can click other buttons on the screen without freezing the app.
- `stop_training_session`: This safely pauses the AI if you hit STOP, making sure no files get corrupted or broken.

---

## 2. The Video Testing Script (`video_detection.py`)

This file is responsible for testing your videos.

**What it does:**
- `extract_frames`: Video files are just thousands of pictures played very fast. This script chops your video up into individual pictures so the AI can look at them easily. To save time, you can tell it to skip pictures (like only checking every 5th picture).
- `detect_video_frames`: This hands every chopped picture to the AI and asks, "Where is the crack?"
- `create_annotated_video`: Once the AI colors the cracks, this script glues all the chopped pictures back together into a brand new video for you to watch!

---

## 3. The Sync Script (`sync.py`)

This file is responsible for finding lost models.

**What it does:**
- `discover_offline_runs`: If your friend teaches an AI on their computer and emails you the folder, this script notices when you drop it in the app's folder. It automatically reads the files and adds the model to your history so you can see their scores.
