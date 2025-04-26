import streamlit as st
import cv2
import numpy as np
import os

# Paths for YOLO model files
weights_path = "yolov3-tiny.weights"
cfg_path = "yolov3.cfg"

# Check if the YOLO files exist in the current directory
if not os.path.exists(weights_path):
    st.error(f"File {weights_path} not found in the repository!")
elif not os.path.exists(cfg_path):
    st.error(f"File {cfg_path} not found in the repository!")
else:
    # Load the YOLO model
    try:
        net = cv2.dnn.readNet(weights_path, cfg_path)
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        st.success("YOLO model loaded successfully!")

        # To process image input
        image = st.file_uploader("Upload Image for Object Detection", type="jpg")
        if image is not None:
            # Convert uploaded image to OpenCV format
            img = np.array(bytearray(image.read()), dtype=np.uint8)
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)

            # Perform object detection
            blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)

            # Process outputs and display results
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
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            # Apply Non-Maximum Suppression (NMS) to remove redundant boxes
            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            # Draw bounding boxes on the image
            if len(indices) > 0:
                for i in indices.flatten():
                    x, y, w, h = boxes[i]
                    label = str(class_ids[i])
                    confidence = str(round(confidences[i], 2))
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(img, label + ":" + confidence, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # Display the result on Streamlit
            st.image(img, channels="BGR")

    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
