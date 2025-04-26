import streamlit as st
import numpy as np
import cv2
import os
from darknet import Darknet
import torch

# Paths for YOLO model files
weights_path = "yolov3-tiny.weights"
cfg_path = "yolov3.cfg"

# Check if the YOLO files exist in the current directory
if not os.path.exists(weights_path):
    st.error(f"File {weights_path} not found in the repository!")
elif not os.path.exists(cfg_path):
    st.error(f"File {cfg_path} not found in the repository!")
else:
    try:
        # Load the YOLO model using Darknet
        net = Darknet(cfg_path)
        net.load_weights(weights_path)

        st.success("YOLO model loaded successfully!")

        # To process image input
        image = st.file_uploader("Upload Image for Object Detection", type="jpg")
        if image is not None:
            # Convert uploaded image to OpenCV format
            img = np.array(bytearray(image.read()), dtype=np.uint8)
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)

            # Perform object detection using Darknet
            detections = net.detect(img)

            # Draw bounding boxes on the image
            for detection in detections:
                label, confidence, bbox = detection
                x, y, w, h = bbox
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, f"{label}: {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # Display the result on Streamlit
            st.image(img, channels="BGR")

    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
