from fastapi import FastAPI, File, UploadFile, HTTPException, Response
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
import cv2
import numpy as np
from ultralytics import YOLO, solutions
import os
import tempfile
import asyncio
from object_detection import process_image, gen_frames , process_video

# Initialize FastAPI app
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Initialize YOLO model
yolo_model_path = "yolov10s.pt"
yolo_model = YOLO(yolo_model_path)
classes_names = yolo_model.names

class CustomObjectCounter(solutions.ObjectCounter):
    def __init__(self, classes_names):
        super().__init__(
            view_img=False,
            reg_pts=[],
            names=classes_names,
            draw_tracks=False,
            line_thickness=2,
        )

# Initialize counter with YOLO model names
counter = CustomObjectCounter(classes_names)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process_image")
async def process_image_route(image: UploadFile = File(...)):
    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Process the image
    processed_img = process_image(yolo_model, counter, img)

    # Encode processed image to JPEG format
    _, jpeg = cv2.imencode('.jpg', processed_img)

    return Response(content=jpeg.tobytes(), media_type="image/jpeg")

@app.post("/process_video")
async def process_video_route(video: UploadFile = File(...)):
    # Save the uploaded video temporarily
    with tempfile.NamedTemporaryFile(delete=True) as temp_video:
        temp_video.write(await video.read())
        temp_video_path = temp_video.name
    
        # Process the video frames
        frames = process_video(yolo_model, counter, temp_video_path)

    # Return a StreamingResponse with the processed frames
    async def generate():
        for frame in frames:
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
    
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


# Start RTSP video feed endpoint (if needed)
@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(gen_frames(yolo_model, counter), media_type="multipart/x-mixed-replace; boundary=frame")

# Main function to run the FastAPI application
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
