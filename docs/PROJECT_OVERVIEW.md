# Crack Detection App Overview

## The Problem

Large buildings, bridges, and big machines often suffer from cracks over time due to weather and stress. Finding and fixing these cracks usually means sending a human to inspect them by hand. This can be slow, dangerous, and sometimes people miss small details. 

If cracks are missed, machines and buildings can break unexpectedly, costing a lot of money and causing delays.

## The Solution

We created an easy-to-use smart laboratory application. This app uses Artificial Intelligence (AI) to automatically look at pictures or videos and find cracks in real-time. It draws clear, colored outlines over the damage so you know exactly where the problems are.

## How the AI Learns (Data)

To make the AI smart, we feed it pictures of cracks. These pictures are called "Datasets".
- The AI requires a specific folder structure (called the YOLO layout).
- You can find free pictures of cracks from public websites, or add your own private photos from drones and cameras.
- The app organizes these pictures so the AI knows exactly where the cracks are drawn.

## The AI Brain (Model Training)

We use a very fast and accurate AI model called **YOLO Segmentation**. It's designed to run quickly, even on small computers like a Raspberry Pi. 

The AI looks at the images you give it and learns over multiple rounds (called Epochs). You can monitor this learning process directly from the app interface without needing to understand code (see [DOCUMENTATION.md](../DOCUMENTATION.md)).

## The Application Interface

The app is designed to look like a vintage science laboratory from the 1970s. It makes complex tasks easy:
- **Datasets**: Easily load pictures into the app.
- **Training**: Teach the AI with a single click and watch it learn on live charts.
- **Models**: Check your past AI brains and their scores.
- **Detection**: Upload a new picture or video and let the AI find the cracks for you.

## Looking Forward

In the future, we plan to:
- Make the app run entirely on small drones for instant live checking.
- Teach the AI to detect other issues like rust and peeling paint.
- Add cloud features so teams can share AI results online.

## Questions?

If you have any questions, please check the [FAQ.md](FAQ.md).
