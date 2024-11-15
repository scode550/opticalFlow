import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
from cv2 import VideoCapture, cvtColor, COLOR_BGR2RGB

# Load YOLOv8 Nano model (yolov8n) using the ultralytics package
@st.cache_resource
def load_model():
    # Load the YOLOv8 nano model using the ultralytics package
    model = YOLO('yolov11n.pt')  # You can specify a local model path or a URL
    return model

# Set up webcam capture
def capture_video():
    cap = VideoCapture(0)  # Use default webcam (change index if necessary)
    return cap

# Run object detection on the frame
def detect_objects(frame, model):
    # Convert the frame from BGR (OpenCV format) to RGB (YOLO model expects RGB input)
    img_rgb = cvtColor(frame, COLOR_BGR2RGB)
    
    # Perform inference using YOLOv8 model
    results = model(img_rgb)  # Detect objects in the image
    
    # Render results (draw bounding boxes on the image)
    return results

# Calculate optical flow between two frames
def calculate_optical_flow(prev_gray, curr_gray):
    # Calculate optical flow using Farneback method
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow

# Estimate distance based on optical flow (simple heuristic)
def estimate_distance(flow, object_box):
    # Simple heuristic: larger flow vectors indicate closer objects
    # Get the center of the bounding box
    center_x = int((object_box[0] + object_box[2]) / 2)
    center_y = int((object_box[1] + object_box[3]) / 2)
    
    # Look at the average flow at the center of the bounding box
    flow_at_center = flow[center_y, center_x]
    distance_estimate = np.linalg.norm(flow_at_center)  # magnitude of the flow vector
    
    return distance_estimate

# Display video stream and inference results
def display_video(cap, model):
    st.title("YOLOv8 Nano Object Detection with Optical Flow and Distance Estimation")

    # Create a placeholder for the video stream
    frame_placeholder = st.empty()

    # Previous frame for optical flow
    prev_gray = None
    prev_bboxes = None  # Previous bounding boxes
    prev_objects = []  # Store the object info from previous frame

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to grab frame.")
            break
        
        # Convert to grayscale for optical flow calculation
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Perform object detection
        results = detect_objects(frame, model)

        # Get the image with bounding boxes drawn
        img_with_boxes = np.array(results.render()[0])  # Rendered image with bounding boxes
        
        # Calculate optical flow if we have a previous frame
        if prev_gray is not None:
            flow = calculate_optical_flow(prev_gray, curr_gray)

            # Estimate distances for detected objects
            for i, bbox in enumerate(results.xywh[0].cpu().numpy()):
                # bbox = [x_center, y_center, width, height]
                x, y, w, h = bbox
                object_box = [x - w/2, y - h/2, x + w/2, y + h/2]  # Convert to [x1, y1, x2, y2]

                # Estimate distance based on optical flow
                distance = estimate_distance(flow, object_box)

                # Draw distance on the image (for visual feedback)
                cv2.putText(img_with_boxes, f'Dist: {distance:.2f}', (int(x), int(y)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the image with bounding boxes and distance
        img_pil = Image.fromarray(img_with_boxes)
        frame_placeholder.image(img_pil, use_column_width=True)

        # Add a button to stop the webcam stream
        if st.button('Stop'):
            cap.release()
            break

        # Update the previous frame and previous bounding boxes
        prev_gray = curr_gray
        prev_bboxes = results.xywh[0].cpu().numpy()

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
