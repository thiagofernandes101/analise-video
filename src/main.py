import cv2 as cv
import numpy as np
import os

script_directory = os.path.dirname(__file__)
video_path = '/home/thiagofernandes101/projects/fiap/analise-video/videos/Unlocking Facial Recognition_ Diverse Activities Analysis.mp4'
proto_path = os.path.join(script_directory, "deploy.prototxt")
model_path = os.path.join(script_directory, "res10_300x300_ssd_iter_140000.caffemodel")
net = cv.dnn.readNetFromCaffe(proto_path, model_path)

video_capture = cv.VideoCapture(video_path)

while True:
    ret, frame = video_capture.read()
    
    if not ret:
        print("End of video stream or cannot read the video.")
        break
    
    (h, w) = frame.shape[:2]
    blob = cv.dnn.blobFromImage(
        cv.resize(frame, (300, 300)), 
        1.0,
        (300, 300), 
        (104.0, 177.0, 123.0)
    )
    
    net.setInput(blob)
    detections = net.forward()
    
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            cv.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
        
    cv.imshow('Face Detection on Video', frame)
    
    if cv.waitKey(1) == ord('q'):
        break

video_capture.release()
cv.destroyAllWindows()