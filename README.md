# HumanTrack_AI
Real-Time Human Detection and Counting with FastAPI and YOLO
Setup
Prerequisites
Python 3.7+
Pipenv or virtualenv for Python environment management


Install dependencies:

pip install -r requirements.txt

Running the Application
Set up the environment variables:

Replace 'yolov10s.pt' with your actual YOLO model path if necessary
export model_path='yolov10s.pt'

Start the FastAPI server:

uvicorn app:app --host 0.0.0.0 --port 8001

Access the application at http://localhost:8001.

Usage
Endpoints
/: Web interface to upload files and view processed results.
/process_image: POST endpoint to upload an image for object detection.
/process_video: POST endpoint to upload a video file for object detection.
/video_feed: GET endpoint to stream a live RTSP video feed with object detection.
How to Run
Access the web interface by navigating to http://localhost:8001 in your browser.

Upload an image or video file for processing using the respective endpoints.

View the processed results directly on the web interface.

Notes
Ensure proper configuration of YOLO model path using the model_path environment variable.
The application supports real-time video processing and streaming via RTSP feeds.
Additional Resources
FastAPI Documentation: https://fastapi.tiangolo.com/
YOLO Object Detection: https://pjreddie.com/darknet/yolo/
Ultralytics YOLO: https://docs.ultralytics.com/models/yolov10/
OpenCV Documentation: https://docs.opencv.org/
