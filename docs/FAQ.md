# FAQ - Frequently Asked Questions

1. [How do I add a new dataset of images?](#how-do-i-add-a-new-dataset-of-images)
2. [How should my folders and files be organized?](#how-should-my-folders-and-files-be-organized)
3. [How do I teach the AI with more images?](#how-do-i-teach-the-ai-with-more-images)
4. [What happens if the training crashes or stops?](#what-happens-if-the-training-crashes-or-stops)
5. [Can I test the AI on videos?](#can-i-test-the-ai-on-videos)
6. [Can I bring in a model trained by someone else?](#can-i-bring-in-a-model-trained-by-someone-else)

---

## How do I add a new dataset of images?
The app needs your images to be on the same computer where the app is running.
- Put your image folder on your computer.
- Click on the **Datasets** tab in the app.
- Type in the location of your folder (like `./datasets/my-cracks/`).
- Click **Import**. The app will do the rest of the work and organize your images for the AI automatically.

## How should my folders and files be organized?
The app requires your folders to follow the standard **YOLO format**. 

**Here is what your folder should look like:**
```text
my_dataset_folder/
├── data.yaml
├── train/
│   ├── images/  (Put training pictures here)
│   └── labels/  (Put training label text files here)
└── val/
    ├── images/
    └── labels/
```

**Label format:**
- Every image should have a text file (`.txt`) with the exact same name.
- If an image doesn't have any cracks in it, you don't need a `.txt` file at all. The AI will learn that this image is completely safe.

## How do I teach the AI with more images?
- Gather new pictures of cracks and draw outlines around them using an annotation tool (save them in the YOLO format).
- Update your dataset folder on the computer, then click import on the **Datasets** tab.
- Go to the **Training** tab.
- Choose a model.
- Click **Start Training** to watch the app update its learning in real-time!

## What happens if the training crashes or stops?
Don't worry! 
- If your computer turns off or the training stops unexpectedly, restart the app.
- The previous training session is saved securely.
- Go to the **Training** tab, find your stopped session, and click **Resume**.
- The AI will pick up right where it left off!

## Can I test the AI on videos?
Yes! You don't need any other software.
- Go to the **Detection** tab and choose to upload a Video.
- Pick how many frames to skip to make processing faster (this is called the Sample Interval).
- The app will grab frames from the video, find the cracks, color them, and stitch the video back together so you can watch the result. 

## Can I bring in a model trained by someone else?
Yes! 
- If someone else trained a model, simply ask them for the result folder (it usually contains files like `results.csv`, `args.yaml` and a `.pt` weights file).
- Drop this folder into the `./checkpoints` directory on your computer.
- Open the app, and it will automatically find the new model and add it to your history.
