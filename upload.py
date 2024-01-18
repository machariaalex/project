import streamlit as st
import cv2
import numpy as np
import time
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os

# Initialize Streamlit app
st.title("Real-time Video Summarization System")

# Ask the user to choose 'Realtime' or 'Upload'
option = st.radio("Select an option:", ("Realtime", "Upload"))

# Video file upload logic
if option == "Upload":
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi"])
    
    if uploaded_file is not None:
        # Save the uploaded video file
        video_path = "uploaded_video.mp4"
        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())
        st.success("Video uploaded successfully!")
else:
    video_path = 0  # Use webcam (index 0)

# Create VideoWriters for both original and summarized videos
width, height = 640, 480
original_writer = cv2.VideoWriter('orig_video.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 25, (width, height))
summarized_writer = cv2.VideoWriter('summarized_video.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 25, (width, height))

# Set the duration of video summarization (90 seconds)
duration = 90  # seconds
end_time = time.time() + duration

# Create VideoWriter for recording events
event_writer = None

# Virtual fence parameters
fence_line_x = int(0.2 * width)  # X-coordinate for the virtual fence line
fence_color = (0, 255, 0)  # Green color for the fence line
fence_thickness = 2  # Thickness of the fence line

# Email configuration (replace with your own details)
smtp_server = "smtp.gmail.com"
smtp_port = 587  # Adjust if needed
smtp_username = "machariaalex456@gmail.com"
smtp_password = "htlm qbby meqi sqwy"
recipient_email = "cheronotruphena66@gmail.com"  # Replace with the recipient's email address

def send_email(subject, body):
    # Setup the MIME
    message = MIMEMultipart()
    message['From'] = smtp_username
    message['To'] = recipient_email
    message['Subject'] = subject

    # Attach the message body
    message.attach(MIMEText(body, 'plain'))

    # Connect to the SMTP server
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        # Login to the email account
        server.starttls()  # Use if your server requires a secure connection
        server.login(smtp_username, smtp_password)

        # Send the email
        server.sendmail(smtp_username, recipient_email, message.as_string())

# Initialize variables
ret, frame1 = cv2.VideoCapture(video_path).read()
prev_frame = frame1
a = 0
b = 0
c = 0
object_count = 0
people_count = 0
objects_in_motion = set()

# Load MobileNet SSD model
protopath = 'MobileNetSSD_deploy.prototxt'
modelpath = 'MobileNetSSD_deploy.caffemodel'
detector = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)

# Define object classes
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Main loop
while time.time() < end_time:
    # Read a frame from the camera or uploaded video
    ret, frame = cv2.VideoCapture(video_path).read()

    # Perform object detection
    (H, W) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
    detector.setInput(blob)
    person_detections = detector.forward()

    # Draw virtual fence line
    cv2.line(frame, (fence_line_x, 0), (fence_line_x, height), fence_color, fence_thickness)

    # Process detected objects
    for i in np.arange(0, person_detections.shape[2]):
        confidence = person_detections[0, 0, i, 2]
        if confidence > 0.5:
            idx = int(person_detections[0, 0, i, 1])

            # Increment object_count when a person is detected
                        # Increment object_count when a person is detected
            if CLASSES[idx] == "person":
                people_count += 1
            else:
                objects_in_motion.add(CLASSES[idx])

            person_box = person_detections[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = person_box.astype("int")

            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)

            # Save original frame to original video
            original_writer.write(frame)

            # Save summarized frame to summarized video
            if np.sum(np.absolute(frame - prev_frame)) / np.size(frame) > 20.:
                summarized_writer.write(frame)
                prev_frame = frame
                a += 1
            else:
                b += 1

            # Save original frame to event video
            if event_writer is None:
                event_writer = cv2.VideoWriter(f'event_{time.time()}.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 25, (width, height))

            event_writer.write(frame)

            # Check if the person/object crossed the virtual fence
            if startX < fence_line_x < endX:
                # Save an image of the trespasser as .png
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_filename = f'trespasser_{timestamp}.png'
                cv2.imwrite(image_filename, frame)

                # Notify about the trespasser
                send_email("Trespasser Alert", f"Someone has trespassed your premise. Image attached: {image_filename}")

    # If there was no person detected, close the event_writer
    if event_writer is not None and people_count == 0:
        event_writer.release()
        event_writer = None

    # Display the frame using Streamlit
    st.image(frame, channels="BGR")

    c += 1

    # Check for the 'q' key to exit the loop
    if st.button("Stop Summarization"):
        break

# Release resources after the loop ends
cv2.destroyAllWindows()

# Save people count and objects in motion to a text file
with open('activity_summary.txt', 'w') as file:
    file.write(f"People count: {people_count}\n")
    file.write("Objects in motion:\n")
    for obj in objects_in_motion:
        file.write(f"{obj}\n")

# Print statistics
st.write("Total frames: ", c)
st.write("People count: ", people_count)
st.write("Objects in motion: ", objects_in_motion)

# Provide download links for the summarized video and activity summary
st.markdown(f"[Download Summarized Video](summarized_video.mp4)")
st.markdown(f"[Download Activity Summary](activity_summary.txt)")
