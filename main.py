import streamlit as st
import cv2
import torch
from PIL import Image
import numpy as np

# Load YOLOv8 Nano model (yolov8n) from the ultralytics repository
@st.cache_resource
def load_model():
    # Load the YOLOv8 nano model using the ultralytics package
    model = torch.hub.load('ultralytics/yolov8', 'yolov8n')  # yolov8n is the nano version of YOLOv8
    return model

# Set up webcam capture
def capture_video():
    cap = cv2.VideoCapture(0)  # Use default webcam (change index if necessary)
    return cap

# Run object detection on the frame
def detect_objects(frame, model):
    # Convert the frame from BGR (OpenCV format) to RGB (YOLO model expects RGB input)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Perform inference using YOLOv8 model
    results = model(img_rgb)  # Detect objects in the image
    
    # Render results (draw bounding boxes on the image)
    return results

# Display video stream and inference results
def display_video(cap, model):
    st.title("YOLOv8 Nano Object Detection in Streamlit")

    # Create a placeholder for the video stream
    frame_placeholder = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to grab frame.")
            break
        
        # Perform object detection
        results = detect_objects(frame, model)

        # Get the image with bounding boxes drawn
        img_with_boxes = np.array(results.render()[0])  # Rendered image with bounding boxes

        # Convert the image with bounding boxes to PIL format for Streamlit
        img_pil = Image.fromarray(img_with_boxes)

        # Display the image in the Streamlit app
        frame_placeholder.image(img_pil, use_column_width=True)

        # Add a button to stop the webcam stream
        if st.button('Stop'):
            cap.release()
            break

# Main function to run the Streamlit app
def main():
    # Load the YOLOv8 Nano model
    model = load_model()

    # Set up video capture (default webcam)
    cap = capture_video()

    # Display the video stream and run object detection
    display_video(cap, model)

    # Release the camera when done
    cap.release()

if __name__ == "__main__":
    main()
