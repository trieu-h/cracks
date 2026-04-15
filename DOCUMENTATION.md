# Crack Detection Laboratory - Simple Guide

## Table of Contents

1. [What is this?](#what-is-this)
2. [What runs the app?](#what-runs-the-app)
3. [User Guide](#user-guide)
4. [Troubleshooting Help](#troubleshooting-help)

---

## What is this?

This is a web application that looks like a vintage 1970s laboratory. It is built to help anyone easily teach an Artificial Intelligence (AI) how to find cracks in physical materials, and then use that AI to automatically highlight damage on your own photos and videos.

### Key Features
- **All-in-One system**: You can load pictures, teach the AI, and test the AI all in the same place.
- **Remembers Everything**: All of your training sessions are saved securely. You never lose your progress.
- **Live Charts**: Watch colorful charts move as the AI learns!
- **Fast Video Testing**: Upload a drone video and let the AI find every crack frame by frame.

---

## What runs the app?

Behind the scenes, the app uses modern technology:
- **Backend (The Brain)**: Python helps the app think and save files using a simple database called SQLite.
- **Frontend (The Screen)**: React and TailwindCSS make the app look like a vintage, beautiful laboratory.
- **ML Framework (The AI)**: We use a popular, extremely fast AI called **YOLO** (You Only Look Once) to learn and find the cracks.

---

## User Guide

### 1. Adding Images (Datasets)
To teach the AI, you need a folder filled with pictures of cracks. 
- Go to the **Datasets** tab.
- Type in where your folder is saved on your computer.
- Click **Import** and the app will organize the pictures for you!

### 2. Teaching the AI (Training)
- Go to the **Training** tab.
- Choose your folder of pictures.
- Click **Start Training**! 
- The computer will do all the hard work. You will see a line chart tracking how smart the AI is getting!

### 3. Testing the AI (Detection)
- Did the AI finish learning? Let's check!
- Go to the **Detection** tab.
- Upload any picture or video.
- Click **Run Detection**. The AI will scan your file and color over any cracks it finds.

---

## Troubleshooting Help

### 1. The app says my folder is missing?
Make sure you typed the exact location of your folder. Windows computers and Mac computers write folder locations differently. Ensure the path exactly matches where the folder is on your hard drive.

### 2. The training charts stop moving?
Sometimes a computer runs out of memory while teaching the AI. Check if you have too many programs open. If the app froze, simply restart it. The app saves your place automatically, so you can go to the **Training** tab and click **Resume**!

### 3. Will this work on Mac?
Yes! However, Macs do not have the specific graphics cards (NVIDIAs) needed to train the AI very quickly. It will work, but it might be slower than a heavy-duty Windows computer.
