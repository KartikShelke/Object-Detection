import os
import cv2
import streamlit as st

st.write("Current directory:", os.getcwd())
st.write("Directory contents:", os.listdir())

try:
    net = cv2.dnn.readNet('./yolov3.weights', './yolov3.cfg')
    st.success("Model loaded successfully! 🚀")
except Exception as e:
    st.error(f"Model loading failed: {e}")
