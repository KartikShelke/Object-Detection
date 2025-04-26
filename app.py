import os
import urllib.request
import streamlit as st
import cv2
import numpy as np

# Function to download a file if not present
def download_file(url, filename):
    if not os.path.exists(filename):
        st.info(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)
        st.success(f"Downloaded {filename}!")

# URLs for YOLOv3 weights and config
cfg_url = 'https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg'
weights_url = 'https://pjreddie.com/media/files/yolov3.weights'

# Check and download YOLOv3 files
download_file(cfg_url, 'yolov3.cfg')
download_file(weights_url, 'yolov3.weights')

# Load YOLOv3 model
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

# Get class names
classes = []
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Get output layers
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Streamlit UI
st.title("YOLOv3 Object Detection")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Prepare image for YOLO
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(416, 416), mean=(0,0,0), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Visualize detections
    class_ids = []
    confidences = []
    boxes = []
    height, width, channels = img.shape

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
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

    # Non-Maximum Suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw boxes
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = (255, 0, 0)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    st.image(img, caption='Detected Image', use_column_width=True)
