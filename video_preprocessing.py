import cv2
import numpy as np

# Open the video file
input_video_path = 'Videos/vid2.mp4'
output_video_path = 'Videos/output_video.mp4'

# Capture video from the input file
cap = cv2.VideoCapture(input_video_path)

# Get the original video frame size and fps
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create a VideoWriter object to save the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
out = cv2.VideoWriter(output_video_path, fourcc, fps, (1020, 500), isColor=False)

# Loop through the frames of the video
while True:
    
    ret, frame = cap.read()
    
    if not ret:
        break

    # Get the center of the frame
    center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2
    
    # Crop the frame to 1020x500 size (centered)
    cropped_frame = frame[center_y - 250:center_y + 250, center_x - 510:center_x + 510]

    # Convert the cropped frame to grayscale (one channel)
    gray_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)

    # Write the frame to the output video
    out.write(gray_frame)

    # Optionally, display the frame (for testing purposes)
    cv2.imshow('Cropped and Grayscale Frame', gray_frame)
    
    # Press 'q' to exit the display window early (optional)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything when done
cap.release()
out.release()
cv2.destroyAllWindows()
