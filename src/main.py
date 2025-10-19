import cv2 as cv
import os

script_directory = os.path.dirname(__file__)
video_path = '/home/thiagofernandes101/projects/fiap/analise-video/videos/Unlocking Facial Recognition_ Diverse Activities Analysis.mp4'

video_capture = cv.VideoCapture(video_path)

face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = video_capture.read()
    
    if not ret:
        print("End of video stream or cannot read the video.")
        break
    
    gray_scale_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_scale_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
    cv.imshow('Face Detection on Video', frame)
    
    if cv.waitKey(1) == ord('q'):
        break

video_capture.release()
cv.destroyAllWindows()