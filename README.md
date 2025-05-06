# Image-Stitching-from-a-Video.
This project provides a Python pipeline for image stitching and mapping from video frames, specifically designed for drone-based surveillance. The code automatically extracts frames from a drone video, detects and matches features, estimates homographies, and stitches the images together to generate a large, seamless map of the surveyed area.

# Features
Frame Extraction: Pulls frames from drone video footage.
Image Stitching: Uses SIFT feature detection and RANSAC-based homography estimation to align and merge images.
Panorama Creation: Produces a single, large-scale map from multiple overlapping images.

# How to Run

1. Install Dependencies
Make sure you have Python 3 and the following packages installed:

pip install opencv-python opencv-contrib-python numpy matplotlib scikit-learn

3. Prepare Your Data
Place your drone video in a known location (e.g., input/drone_video.mp4).
Create a folder named data in your project directory for extracted frames.

4. Extract Frames from Video
Edit the script's video path if needed, then run the frame extraction function to save frames as images:
In your script or Python shell

from your_script import extract_video_frames
extract_video_frames('input/drone_video.mp4')
This will save frames to D:\frames\ (edit the path in the script as needed).

6. Place Images for Stitching
Copy or move the extracted frames (or any images you want to stitch) into the data folder.

7. Run the Stitching Script
Run the main script:

python your_script.py

The script will read all images in the data directory, stitch them together, and display the resulting panoramic map.
