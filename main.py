from ultralytics import YOLO
import cv2

# Load yolo8 model
model = YOLO('yolov8n.pt')

# Load video
video_path = './test.mp4'
cap = cv2.VideoCapture(video_path)

# read frames
ret = True
while ret:
    ret, frame = cap.read()
    
    # detect objects
    # track objects
    results = model.track(frame, persist=True)
    
    # plot results
    frame_ = results[0].plot()
    
    # visualize
    cv2.imshow('frame', frame_)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

