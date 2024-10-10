import cv2
import numpy as np

def nothing(x):
    pass

# Create a window
cv2.namedWindow("Tracking")

# Create trackbars for color change
cv2.createTrackbar("LH", "Tracking", 0, 179, nothing)  # Lower Hue
cv2.createTrackbar("LS", "Tracking", 0, 255, nothing)  # Lower Saturation
cv2.createTrackbar("LV", "Tracking", 0, 255, nothing)  # Lower Value
cv2.createTrackbar("UH", "Tracking", 179, 179, nothing)  # Upper Hue
cv2.createTrackbar("US", "Tracking", 255, 255, nothing)  # Upper Saturation
cv2.createTrackbar("UV", "Tracking", 255, 255, nothing)  # Upper Value

# Capture video from file
cap = cv2.VideoCapture('MEJA_6_1.5HZ.mkv')  # Provide the correct file path to your video

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    # Read frame-by-frame from video
    ret, frame = cap.read()

    # Check if the frame is successfully read
    if not ret:
        print("Error: Can't receive frame (stream end?). Exiting ...")
        break

    # Convert the frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get the new values of the trackbars in real time
    lh = cv2.getTrackbarPos("LH", "Tracking")
    ls = cv2.getTrackbarPos("LS", "Tracking")
    lv = cv2.getTrackbarPos("LV", "Tracking")
    uh = cv2.getTrackbarPos("UH", "Tracking")
    us = cv2.getTrackbarPos("US", "Tracking")
    uv = cv2.getTrackbarPos("UV", "Tracking")

    # Define the lower and upper HSV range
    lower_bound = np.array([lh, ls, lv])
    upper_bound = np.array([uh, us, uv])

    # Create a mask for that range
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Perform bitwise-AND on the original image and mask to extract the color region
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Display the result
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    cv2.imshow("Result", result)

    # Exit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
