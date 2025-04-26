import streamlit as st
import cv2
import numpy as np

# Load YOLO model and class labels
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
layer_names = net.getLayerNames()
output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]

# COCO class labels (for YOLOv3)
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Function to detect objects
def detect_objects(image):
    # Convert image to blob
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Initialize lists for detected objects
    class_ids = []
    confidences = []
    boxes = []
    height, width, channels = image.shape

    # Loop over the detections
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # You can adjust the confidence threshold
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maxima suppression to eliminate redundant overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw boxes and labels on the image
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = (0, 255, 0)
            cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return image

# Streamlit app layout
st.title('Real-Time Object Detection with YOLO')

# Open a webcam stream
cap = cv2.VideoCapture(0)

# Display video feed and detect objects in real-time
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.write("Failed to grab frame.")
        break

    # Detect objects in the current frame
    frame_with_objects = detect_objects(frame)

    # Convert frame to RGB for display in Streamlit (OpenCV uses BGR)
    frame_rgb = cv2.cvtColor(frame_with_objects, cv2.COLOR_BGR2RGB)

    # Display the frame in Streamlit
    st.image(frame_rgb, channels="RGB", use_column_width=True)

    # Allow user to stop the camera feed with a button
    if st.button('Stop Camera'):
        cap.release()
        st.write("Camera stopped.")
        break

# Release the camera once done
cap.release()
cv2.destroyAllWindows()
