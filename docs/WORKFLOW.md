# How to Use the App (Step-by-Step)

This guide takes you through the step-by-step process of using the Crack Detection Laboratory. You will learn how to add pictures of cracks, train the AI model, and test it to see the results.

## 1. Add Your Images (Dataset Import)
**Go to the `Datasets` page.**
- First, make sure you have a folder on your computer with your crack pictures and their labels. The folder needs to be in a specific layout (the YOLO format).
- Type in the folder path (for example, `./datasets/crack_data/`).
- Click **Import**. The app reads your folder and saves it into the system so the AI can use it to learn.

## 2. Teach the AI (Launch Training)
**Go to the `Training` page.**
- Once your dataset is loaded, it's time to teach the AI what cracks look like.
- Choose your dataset from the dropdown menu.
- Select one of the available AI models (like YOLO Segmentation).
- Adjust simple settings (like how many times the AI should review the pictures, known as "Epochs").
- Click **Start Training**. The computer will start learning. You will see live charts showing you how the app gets smarter over time!

## 3. View Saved AI Models (Review Checkpoints)
**Go to the `Models` page.**
- After the training finishes, go to the Models page.
- Here, you can find a list of all your fully trained AI models. Have a look at their scores to see how accurate they are.
- Every saved model (checkpoint) is kept safely so you can reuse it later.
- *(Note: If you or a friend trained an AI model on a different computer, just copy the result folder into the backend folder and click 'Sync Local Models' to add it to your list!)*

## 4. Test the AI (Run Detection)
**Go to the `Detection` page to test.**
- Now that your AI learned how to detect cracks, let's see it in action!
- Open the Detection page.
- Upload an image or a video.
- Click **Run Detection**. The AI will analyze your file and draw colored outlines directly over the cracks it finds!
