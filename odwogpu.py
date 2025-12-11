import cv2
import numpy as np

# Load MobileNet-SSD model (COCO trained, includes 'person' class)
# Download files from: https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API
prototxt_path = r"C:\Users\HARSHIL\Downloads\race_car.mp4\prashidhbhai\MobileNetSSD_deploy.prototxt"
model_path = r"C:\Users\HARSHIL\Downloads\race_car.mp4\prashidhbhai\MobileNetSSD_deploy.caffemodel"

# Initialize the DNN net (automatically runs on CPU if no GPU found)
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Open webcam
cap = cv2.VideoCapture(0)

# COCO class labels MobileNet-SSD was trained on
class_names = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    
    # Preprocess frame for the network (300x300 is the model's input size)
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    
    # Run detection (this is the CPU-intensive part)
    detections = net.forward()
    
    # Parse detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Confidence threshold
            class_id = int(detections[0, 0, i, 1])
            if class_id == 15:  # Class ID for 'person'
                # Get bounding box coordinates
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                
                # Draw box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"Person: {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imshow("CPU Person Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()