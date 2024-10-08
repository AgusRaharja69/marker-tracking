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
cap = cv2.VideoCapture('coba8.mkv')
if not cap.isOpened():
    print("Error: Cannot open video file")
    exit()

# Initialize reference points and flag
reference_points = {}
first_frame_processed = False
pattern_size_cm = 10.5
converter_value = {}

# Open CSV file for writing
with open('oscillation_data_5.csv', mode='w', newline='') as csv_file:
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
                height_pixel = np.linalg.norm(c[i] - c[3])
                converter = float(pattern_size_cm/int(height_pixel))
                converter_value[marker_id[0]] = converter
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

                    # Convert pixel deltas to cm using the converter value
                    delta_x_cm = delta_x * converter_value[marker_id[0]]
                    delta_y_cm = delta_y * converter_value[marker_id[0]]

                    current_data.extend([curr_x, curr_y, ref_x, ref_y, round(delta_x_cm,3), round(delta_y_cm,3)])

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
data = pd.read_csv('oscillation_data_5.csv', header=None, names=column_names, on_bad_lines='skip')

# Convert relevant columns to numeric, handling NaN values
data['time'] = pd.to_numeric(data['time'], errors='coerce')
for id in range(3):
    data[f'delta_y_id_{id}'] = pd.to_numeric(data[f'delta_y_id_{id}'], errors='coerce')

# Find the max delta_y for ID 0 and corresponding values for ID 1 and ID 2
max_delta_y_id_0 = data['delta_y_id_0'].max()
max_time = data['time'][data['delta_y_id_0'].idxmax()]

min_delta_y_id_0 = data['delta_y_id_0'].min()
min_time = data['time'][data['delta_y_id_0'].idxmin()]

# Find corresponding delta_y values for ID 1 and ID 2 at the same 'time'
delta_y_id_1_at_max = data.loc[data['time'] == max_time, 'delta_y_id_1'].values
delta_y_id_2_at_max = data.loc[data['time'] == max_time, 'delta_y_id_2'].values

delta_y_id_1_at_min = data.loc[data['time'] == min_time, 'delta_y_id_1'].values
delta_y_id_2_at_min = data.loc[data['time'] == min_time, 'delta_y_id_2'].values

# Check if any results were found
if delta_y_id_1_at_max.size > 0:
    delta_y_id_1_at_max = delta_y_id_1_at_max[0]
else:
    delta_y_id_1_at_max = None

if delta_y_id_2_at_max.size > 0:
    delta_y_id_2_at_max = delta_y_id_2_at_max[0]
else:
    delta_y_id_2_at_max = None

# Check if any results were found
if delta_y_id_1_at_min.size > 0:
    delta_y_id_1_at_min = delta_y_id_1_at_min[0]
else:
    delta_y_id_1_at_min = None

if delta_y_id_2_at_min.size > 0:
    delta_y_id_2_at_min = delta_y_id_2_at_min[0]
else:
    delta_y_id_2_at_min = None

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
             f'Max Delta Y ID 0: {max_delta_y_id_0:.2f}cm\n'
             f'Delta Y ID 1: {delta_y_id_1_at_max:.2f}cm\n'
             f'Delta Y ID 2: {delta_y_id_2_at_max:.2f}cm' ,
             color='black', fontsize=9, verticalalignment='bottom')

if delta_y_id_1_at_min is not None and delta_y_id_2_at_min is not None:
    plt.text(min_time, min_delta_y_id_0,
             f'min Delta Y ID 0: {min_delta_y_id_0:.2f}cm\n'
             f'Delta Y ID 1: {delta_y_id_1_at_min:.2f}cm\n'
             f'Delta Y ID 2: {delta_y_id_2_at_min:.2f}cm' ,
             color='black', fontsize=9, verticalalignment='bottom')

# abs_min = abs(min_delta_y_id_0)

# if abs_min > max_delta_y_id_0:
#     max_y_0 = abs_min
#     y_1 = abs(delta_y_id_1_at_min)
#     y_2 = abs(delta_y_id_2_at_min)
#     timeData = min_time
# else:
#     max_y_0 = max_delta_y_id_0
#     y_1 = abs(delta_y_id_1_at_max)
#     y_2 = abs(delta_y_id_2_at_max)
#     timeData = max_time

# # Final plot settings
# plt.legend()
# plt.title(f'Combined Oscillation of Markers\n'
#           f'Max Delta Y ID 0: {max_y_0:.2f}cm at time {timeData:.2f}seconds,\n'
#           f'Delta Y ID 1: {y_1:.2f}cm, Delta Y ID 2: {y_2:.2f}cm')


# Final plot settings
plt.legend()
plt.title(f'Combined Oscillation of Markers\n'
          f'Max Delta Y ID 0: {max_delta_y_id_0:.2f}cm at time {max_time:.2f}seconds, Delta Y ID 1: {delta_y_id_1_at_max:.2f}cm, Delta Y ID 2: {delta_y_id_2_at_max:.2f}cm\n'
          f'Min Delta Y ID 0: {min_delta_y_id_0:.2f}cm at time {min_time:.2f}seconds, Delta Y ID 1: {delta_y_id_1_at_min:.2f}cm, Delta Y ID 2: {delta_y_id_2_at_min:.2f}cm')
plt.xlabel('Time (s)')
plt.ylabel('Oscillation (pixels)')
plt.savefig('combined_oscillation_plot_5.png')
plt.show()