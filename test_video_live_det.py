import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('C:/Users/Vanshika/OneDrive/Desktop/Golf_Pose_Det_YOLO/best.pt') 

# Open the webcam or any video source (0 is the default webcam)
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("video_path")

if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # Perform detection on the frame
    results = model(frame)

    # Annotate the frame with bounding boxes and labels
    annotated_frame = results[0].plot()

    # Display the frame with annotations
    cv2.imshow('Golf Pose Video/Live Detection', annotated_frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
