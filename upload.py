 cv2.VideoWriter('orig_video.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 25, (width, height))
    summarized_writer = cv2.VideoWriter('summarized_video.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 25, (width, height))

    # Virtual fence parameters
    fence_line_x = int(0.2 * width)  # X-coordinate for the virtual fence line
    fence_color = (0, 255, 0)  # Green color for the fence line
    fence_thickness = 2  # Thickness of the fence line

    # Initialize variables
    ret, frame1 = video.read()
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
    while True:
        # Read a frame from the uploaded video
        ret, frame = video.read()

        if not ret:
            # End of video
            break

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

        # Display the frame
        st.image(frame, channels="BGR")
        c += 1

    # Release resources after the loop ends
    video.release()
    original_writer.release()
    summarized_writer.release()

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

# Streamlit UI
st.title("Real-time Video Summarization System")

# Ask the user whether to run real-time or upload summarization
choice = st.radio("Select Summarization Option:", ("Real-time", "Upload"))
if choice == "Real-time":
    st.write("Webcam is opening...")
    run_realtime_summarization()
else:
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4"])
    if uploaded_file is not None:
        run_upload_summarization(uploaded_file)
