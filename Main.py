import cv2
import numpy as np
import streamlit as st

def camshift_tracking(frame):
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define a lower and upper color range in HSV for the object you want to track
    lower = np.array([0, 50, 50])
    upper = np.array([10, 255, 255])
    
    # Threshold the HSV image to get only the desired color range
    mask = cv2.inRange(hsv, lower, upper)
    
    # Perform the Camshift algorithm
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    track_box, track_window = cv2.CamShift(mask, (0,0,0,0), term_crit)
    
    # Draw a rectangle around the tracked object
    pts = cv2.boxPoints(track_box)
    pts = np.int0(pts)
    frame = cv2.polylines(frame, [pts], True, (255,0,0), 2)
    
    return frame

# Get the input video
video_file = st.file_uploader("Upload a video", type=["mp4", "avi"])
if video_file is not None:
    cap = cv2.VideoCapture(video_file)

    # Loop over the frames in the video
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frame = camshift_tracking(frame)
            st.image(frame, use_column_width=True)
        else:
            break
    
    cap.release()
