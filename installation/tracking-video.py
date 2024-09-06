import cv2
import csv
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load ArUco marker dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()

# Open video file
cap = cv2.VideoCapture('recorded_video.avi')
if not cap.isOpened():
    print("Error: Cannot open video file")
    exit()

# Initialize reference points and flag
reference_points = {}
first_frame_processed = False

# Open CSV file for writing
with open('oscillation_data.csv', mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    # CSV Header
    csv_writer.writerow(['time', 'x_id_0', 'y_id_0', 'x_ref_id_0', 'y_ref_id_0', 'delta_x_id_0', 'delta_y_id_0',
                         'x_id_1', 'y_id_1', 'x_ref_id_1', 'y_ref_id_1', 'delta_x_id_1', 'delta_y_id_1',
                         'x_id_2', 'y_id_2', 'x_ref_id_2', 'y_ref_id_2', 'delta_x_id_2', 'delta_y_id_2'])

    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or failed to grab frame")
            break

        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect markers
        corners, ids, rejected = cv2.aruco.detectMarkers(gray_frame, aruco_dict, parameters=parameters)

        # Process the first frame
        if not first_frame_processed and ids is not None and len(ids) == 3:
            for i, marker_id in enumerate(ids):
                c = corners[i][0]  # Get corners of the marker
                reference_points[marker_id[0]] = c[0]  # Store first corner as reference point
            first_frame_processed = True

        if first_frame_processed and ids is not None:
            current_data = [time.time() - start_time]  # Capture relative time
            id_0_present = False

            # Loop through detected markers
            for i, marker_id in enumerate(ids):
                c = corners[i][0]
                curr_x, curr_y = int(c[0][0]), int(c[0][1])  # Current corner (top-left)

                if marker_id[0] in reference_points:
                    ref_x, ref_y = reference_points[marker_id[0]]
                    delta_x = curr_x - ref_x
                    delta_y = curr_y - ref_y

                    current_data.extend([curr_x, curr_y, ref_x, ref_y, delta_x, delta_y])

                    if marker_id[0] == 0:
                        id_0_present = True

                    # Visualize the marker positions
                    cv2.circle(frame, (curr_x, curr_y), 2, (0, 255, 0), 2)
                    cv2.circle(frame, (int(ref_x), int(ref_y)), 3, (255, 0, 255), 3)
                else:
                    current_data.extend([None, None, None, None, None, None])

            if id_0_present and len(current_data) == 19:
                csv_writer.writerow(current_data)

        # Display the markers on the frame
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        cv2.imshow('ArUco Marker Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# Correct column names
print("plotting the data ......")
column_names = [
    'time', 'x_id_0', 'y_id_0', 'x_ref_id_0', 'y_ref_id_0', 'delta_x_id_0', 'delta_y_id_0',
    'x_id_1', 'y_id_1', 'x_ref_id_1', 'y_ref_id_1', 'delta_x_id_1', 'delta_y_id_1',
    'x_id_2', 'y_id_2', 'x_ref_id_2', 'y_ref_id_2', 'delta_x_id_2', 'delta_y_id_2'
]

# Load the CSV file with the specified column names
data = pd.read_csv('oscillation_data.csv', header=None, names=column_names, on_bad_lines='skip')

# Convert relevant columns to numeric, handling NaN values
data['time'] = pd.to_numeric(data['time'], errors='coerce')
for id in range(3):
    data[f'delta_y_id_{id}'] = pd.to_numeric(data[f'delta_y_id_{id}'], errors='coerce')

# Find the max delta_y for ID 0 and corresponding values for ID 1 and ID 2
max_delta_y_id_0 = data['delta_y_id_0'].max()
max_time = data['time'][data['delta_y_id_0'].idxmax()]

# Find corresponding delta_y values for ID 1 and ID 2 at the same 'time'
delta_y_id_1_at_max = data.loc[data['time'] == max_time, 'delta_y_id_1'].values
delta_y_id_2_at_max = data.loc[data['time'] == max_time, 'delta_y_id_2'].values

# Check if any results were found
if delta_y_id_1_at_max.size > 0:
    delta_y_id_1_at_max = delta_y_id_1_at_max[0]
else:
    delta_y_id_1_at_max = None

if delta_y_id_2_at_max.size > 0:
    delta_y_id_2_at_max = delta_y_id_2_at_max[0]
else:
    delta_y_id_2_at_max = None

# ---- Plotting each delta_y_id ----
plt.figure(figsize=(18, 12))

# Plot for delta_y_id_0
plt.subplot(3, 1, 1)
plt.plot(data['time'], data['delta_y_id_0'], label='Delta Y ID 0', marker='o', linestyle='--')
plt.title('Delta Y ID 0')
plt.xlabel('Time (s)')
plt.ylabel('Delta Y ID 0')

# Plot for delta_y_id_1
plt.subplot(3, 1, 2)
plt.plot(data['time'], data['delta_y_id_1'], label='Delta Y ID 1', marker='^', linestyle='--', color='green')
plt.title('Delta Y ID 1')
plt.xlabel('Time (s)')
plt.ylabel('Delta Y ID 1')

# Plot for delta_y_id_2
plt.subplot(3, 1, 3)
plt.plot(data['time'], data['delta_y_id_2'], label='Delta Y ID 2', marker='s', linestyle='--', color='red')
plt.title('Delta Y ID 2')
plt.xlabel('Time (s)')
plt.ylabel('Delta Y ID 2')

plt.tight_layout()
plt.savefig('ocillation_plot.png')
plt.show()

# ---- Combined plot with max value ----
plt.figure(figsize=(15, 10))

# Plot for ID 0
plt.plot(data['time'], data['delta_y_id_0'], label='Delta Y ID 0', marker='o', linestyle='--')

# Plot for ID 1
plt.plot(data['time'], data['delta_y_id_1'], label='Delta Y ID 1', marker='^', linestyle='--')

# Plot for ID 2
plt.plot(data['time'], data['delta_y_id_2'], label='Delta Y ID 2', marker='s', linestyle='--', color='red')

# Add text annotations for the max values
if delta_y_id_1_at_max is not None and delta_y_id_2_at_max is not None:
    plt.text(max_time, max_delta_y_id_0,
             f'Max Delta Y ID 0: {max_delta_y_id_0:.2f}\n'
             f'Delta Y ID 1: {delta_y_id_1_at_max:.2f}\n'
             f'Delta Y ID 2: {delta_y_id_2_at_max:.2f}',
             color='black', fontsize=9, verticalalignment='bottom')

# Final plot settings
plt.legend()
plt.title(f'Combined Oscillation of Markers\n'
          f'Max Delta Y ID 0: {max_delta_y_id_0:.2f} at time {max_time:.2f}\n'
          f'Delta Y ID 1: {delta_y_id_1_at_max:.2f}\n'
          f'Delta Y ID 2: {delta_y_id_2_at_max:.2f}')
plt.xlabel('Time (s)')
plt.ylabel('Oscillation (pixels)')
plt.savefig('combined_oscillation_plot.png')
plt.show()