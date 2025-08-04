from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, Response
import cv2
import numpy as np
from ultralytics import YOLO, solutions
import os
import asyncio

line_points = [(20, 400), (1080, 400)]
class CustomObjectCounter(solutions.ObjectCounter):
    def __init__(self, classes_names):
        super().__init__(
            view_img=False,
            reg_pts=line_points,
            classes_names=classes_names,
            draw_tracks=True,
            line_thickness=2,
        )

def initialize_yolo_model(model_path):
    return YOLO(model_path)

def process_image(yolo_model, counter, image):
    model = yolo_model
    # Example classes to count
    classes_to_count = [0]  # Example classes: 0 for person
    
    # Track objects in the image
    tracks = model.track(image, persist=False, show=False, classes=classes_to_count)
    # for track in tracks:
    #     track[5] = int(track[5])
    # import pdb 
    # pdb.set_trace()
    # Process image and count objects
    image = counter.start_counting(image, tracks)
    
    return image

rtsp_url = "rtsp://faizan:10412989@192.168.75.94:554/stream1"
async def gen_frames(yolo_model,counter):
    camera = cv2.VideoCapture(rtsp_url)
    if camera.isOpened():
        print("Stream opened successfully")
    else:
        print("Failed to open stream")
    
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        else:
            # Process frame with YOLO model and count objects
            classes_to_count = [0]  # Example: 0 for person
            tracks = yolo_model.track(frame, persist=False, show=False, classes=classes_to_count)
            processed_frame = counter.start_counting(frame, tracks)

            # Encode the processed frame as JPEG
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        await asyncio.sleep(0.01)  



def process_video(yolo_model, counter, video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Error opening video file")

    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame with YOLO model and count objects
        classes_to_count = [0]  # Example: 0 for person
        tracks = yolo_model.track(frame, persist=False, show=False, classes=classes_to_count)
        processed_frame = counter.start_counting(frame, tracks)
        
        # Encode processed frame as JPEG
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()
        frames.append(frame_bytes)

    cap.release()
    cv2.destroyAllWindows()
    
    return frames