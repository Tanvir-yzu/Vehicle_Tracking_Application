import cv2
import pandas as pd
import time
from ultralytics import YOLO
from tracker import *

# Load YOLOv8 model
model = YOLO('yolov8s.pt')

# Function to track the mouse movement and print the BGR values (optional for your task)
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture('cars.mp4')

# Load the COCO class names
with open("coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

# Initialize variables
count = 0
tracker = Tracker()
cy1 = 322  # Position for the counting line
offset = 6  # Offset for the counting threshold
unique_cars = set()  # Set to store unique car IDs

# Get the width and height of the video
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Region of Interest (ROI)
roi_y1, roi_y2 = 200, 500  # Define a region to track cars

# Create video writer to save the output
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (width, height))

# To calculate FPS
prev_frame_time = 0
new_frame_time = 0

# Vehicle count
car_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    count += 1
    if count % 3 != 0:
        continue
    
    frame = cv2.resize(frame, (1020, 500))
    
    # Calculate FPS
    new_frame_time = time.time()
    fps = int(1 / (new_frame_time - prev_frame_time))
    prev_frame_time = new_frame_time
    
    # Draw ROI (Region of Interest) for tracking
    roi_frame = frame[roi_y1:roi_y2, 0:1020]
    
    # YOLO model prediction
    results = model.predict(roi_frame)
    
    # Extract the detection results
    if len(results) > 0 and len(results[0].boxes.data) > 0:
        detection_data = results[0].boxes.data.cpu().numpy()  # Ensure it's a numpy array
        px = pd.DataFrame(detection_data).astype(float)
    
        car_list = []

        # Loop through each detection
        for _, row in px.iterrows():
            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])
            confidence = float(row[4])  # Confidence score of detection
            class_id = int(row[5])  # Class ID from YOLO
            
            # Get the class name based on class_id
            if class_id < len(class_list) and class_id >= 0:
                class_name = class_list[class_id]

                if 'car' in class_name.lower() and confidence > 0.4:  # Filter out low-confidence detections
                    # Append the bounding box adjusted to the original frame coordinates
                    car_list.append([x1, y1 + roi_y1, x2, y2 + roi_y1])

        # Update the tracker with the car bounding boxes
        bbox_id = tracker.update(car_list)
        
        # Draw the tracking information and keep track of unique car IDs
        for bbox in bbox_id:
            x3, y3, x4, y4, car_id = bbox
            cx = (x3 + x4) // 2
            cy = (y3 + y4) // 2
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(frame, str(car_id), (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

            # Add the car ID to the set of unique cars
            unique_cars.add(car_id)

            # Vehicle counting: If the car crosses the counting line (cy1)
            if cy1 - offset < cy < cy1 + offset:
                car_count += 1
                cv2.line(frame, (0, cy1), (1020, cy1), (0, 255, 0), 3)  # Draw green line when car crosses

    # Display the total number of cars detected and FPS
    total_cars_text = f"Total Cars: {len(unique_cars)}"
    cv2.putText(frame, total_cars_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps}", (900, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Crossed Cars: {car_count}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Draw the counting line
    cv2.line(frame, (0, cy1), (1020, cy1), (255, 255, 255), 2)

    # Save the frame to the output video
    out.write(frame)
    
    # Show the frame in a window
    cv2.imshow("RGB", frame)
    
    # Exit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and writer
cap.release()
out.release()
cv2.destroyAllWindows()
