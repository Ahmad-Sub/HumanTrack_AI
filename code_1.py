import cv2

rtsp_url = "rtsp://SarahK:SarahK@192.168.0.10:554/stream2"
camera = cv2.VideoCapture(rtsp_url)