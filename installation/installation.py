import cv2

# Load the dictionary that defines the type of ArUco marker
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# Create detector parameters object
parameters = cv2.aruco.DetectorParameters()

# Open the webcam (camera index 1 for an external camera)
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

# Initialize variables for video recording
recording = False
out = None

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Convert the frame to grayscale (marker detection works better in grayscale)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect markers in the frame
    corners, ids, rejected = cv2.aruco.detectMarkers(gray_frame, aruco_dict, parameters=parameters)

    if recording:
        # Only draw the circles at fixed positions when recording
        height, width, _ = frame.shape
        center_x = width // 2
        center_y = height // 2
        # cv2.circle(frame, (center_x, center_y), 1, (0, 255, 0), 2)
        # cv2.circle(frame, (width - 20, height - 20), 1, (0, 255, 0), 2)
        # cv2.circle(frame, (20, 20), 1, (255, 0, 0), 2)
    else:
        # Draw the detected markers and information if not recording

        height, width, _ = frame.shape
        center_x = width // 2
        center_y = height // 2
        # print(width)
        # print(height)
        cv2.circle(frame, (center_x, center_y), 1, (0, 0, 255), 2)
        cv2.circle(frame, ((width - 90), 90), 1, (0, 0, 255), 2)
        cv2.circle(frame, (90, (height - 90)), 1, (0, 0, 255), 2)
        if ids is not None:
            # Draw the detected marker boundaries
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            # Iterate through all detected markers
            for i, marker_id in enumerate(ids):
                # Get the center of the marker
                c = corners[i][0]
                center_x = int((c[0][0] + c[2][0]) / 2)
                center_y = int((c[0][1] + c[2][1]) / 2)

                # Draw a circle at the center of the marker
                cv2.circle(frame, (center_x, center_y), 10, (0, 255, 0), 2)

                # Display marker ID and coordinates on the frame
                text = f"ID: {marker_id[0]} (x: {center_x}, y: {center_y})"
                cv2.putText(frame, text, (center_x - 50, center_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with markers and IDs
    cv2.imshow('ArUco Marker Tracking', frame)

    # Check for key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit the application
        break
    elif key == ord('r'):  # Start recording
        if not recording:
            recording = True
            # Define the codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter('recorded_video_10Hz.avi', fourcc, 20.0, (frame.shape[1], frame.shape[0]))
            print("Recording started")
    elif key == ord('s'):  # Stop recording
        if recording:
            recording = False
            out.release()
            print("Recording stopped")

    # Write the frame to the video file if recording
    if recording:
        out.write(frame)

# Release the webcam and close windows
cap.release()
if recording:
    out.release()
cv2.destroyAllWindows()
