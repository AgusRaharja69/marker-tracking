import cv2
import numpy as np
import csv
import time

# Define the HSV color range for the object
LH, LS, LV = 0, 231, 32
UH, US, UV = 179, 255, 255
lower_hsv = np.array([LH, LS, LV])
upper_hsv = np.array([UH, US, UV])

# Minimum pixel height for the object
min_height = 150

# Start video capture
cap = cv2.VideoCapture('MEJA_6_5.5HZ.mp4')  # Replace with the video file path

# Variables to store initial position, initial height, and delta movement
initial_position = None
initial_height = None
max_delta_y = -float('inf')
min_delta_y = float('inf')
start_time = time.time()

# Open the CSV file for writing
with open('movement_data_with_cm.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Time", "Delta X (px)", "Delta Y (px)", "Delta Y (cm)"])

    while True:
        # Read a frame
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create a mask for the defined HSV range
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Loop through the contours
        for contour in contours:
            # Get bounding box of the contour
            x, y, w, h = cv2.boundingRect(contour)

            # Filter based on minimum height
            if h >= min_height:
                # Calculate the upper right corner (x + w, y)
                upper_right_x = x + w
                upper_right_y = y

                # Set the initial position and initial height if not set
                if initial_position is None:
                    initial_position = (529, 64)
                    initial_height = h
                    print(initial_position)
                else:
                    # Calculate delta movement from the initial position
                    delta_x = upper_right_x - initial_position[0]
                    delta_y = upper_right_y - initial_position[1]

                    # Track maximum and minimum delta y values
                    if delta_y > max_delta_y:
                        max_delta_y = delta_y
                    if delta_y < min_delta_y:
                        min_delta_y = delta_y

                    # Convert delta_y to cm using the initial height as the scale
                    pixels_per_cm = initial_height / 7.5  # 10 cm corresponds to initial height in pixels
                    delta_y_cm = delta_y / pixels_per_cm

                    # Get the current time
                    current_time = time.time() - start_time

                    # Write the data to the CSV
                    writer.writerow([current_time, delta_x, delta_y, delta_y_cm])

                # Draw the bounding box on the frame
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Tracked Object", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                break  # Only track the first detected object

        # Show the result
        cv2.imshow("Frame", frame)
        cv2.imshow("Mask", mask)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Calculate the max and min delta y in cm using the conversion factor
max_delta_y_cm = max_delta_y / pixels_per_cm
min_delta_y_cm = min_delta_y / pixels_per_cm

# Append max and min delta_y in cm to the CSV file
with open('movement_data_with_cm.csv', mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([])
    writer.writerow(["Initial Height (px)", "Pixels per cm", "Max Delta Y (cm)", "Min Delta Y (cm)"])
    writer.writerow([initial_height, pixels_per_cm, max_delta_y_cm, min_delta_y_cm])

# Release resources
cap.release()
cv2.destroyAllWindows()
