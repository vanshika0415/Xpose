import cv2
from ultralytics import YOLO

# Load the YOLOv8 model (you can choose a different variant like 'yolov8n', 'yolov8s', etc.)
model = YOLO('C:/Users/Vanshika/OneDrive/Desktop/Golf_Pose_Det_YOLO/best.pt')  

# Load an image
image_path = 'C:/Users/Vanshika/OneDrive/Desktop/golf1.jpeg'
image = cv2.imread(image_path)

# Perform detection
results = model(image)

# Display the results
annotated_image = results[0].plot()  # Annotates the image with bounding boxes and labels

# Show the image using OpenCV
cv2.imshow('Golf Pose Image Detection', annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the annotated image
cv2.imwrite('output_image.jpg', annotated_image)
