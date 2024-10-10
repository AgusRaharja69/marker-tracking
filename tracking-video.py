import cv2
import csv
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# Load ArUco marker dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()

# Video paths for processing
tim_name = [9]
meja_no = [1]
ref_x_id0 = None
ref_y_id0 = None

# # Define the video paths and formats
# videos_path = [
#     "videos/TIM_14/1.5HZ/MEJA_6_1.5HZ",
#     "videos/TIM_14/2.5HZ/MEJA_6_2.5HZ",
#     "videos/TIM_14/3.5HZ/MEJA_6_3.5HZ",
#     "videos/TIM_14/4.5HZ/MEJA_6_4.5HZ",
#     "videos/TIM_14/5.5HZ/MEJA_6_5.5HZ"
# ]

# file_formats = [
#     "mp4",
#     "mp4",
#     "mp4",
#     "mkv",
#     "mkv"
# ]


videos_path = []

# Loop through each tim_name and meja_no to construct video paths
for i in range(len(tim_name)):
    for frequency in [1.5, 2.5, 3.5, 4.5, 5.5]:  # Define the frequency options
        video_path = f"videos/TIM_{tim_name[i]}/" \
            f"{frequency}HZ/MEJA_{meja_no[i]}_{frequency}HZ"
        videos_path.append(video_path)

# Function to process each video
def process_video(video_path):
    global ref_x_id0, ref_y_id0
    cap = cv2.VideoCapture(f"{video_path}.mkv")
    
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}.mkv")
        return

    # Get video file name without extension
    video_name = os.path.basename(video_path)  

    # Working freq
    frec_name = video_path.split('/')[2]
    
    # Get the directory of the video
    output_folder = os.path.dirname(video_path)
    
    # Define the paths for the output CSV and plots
    csv_file_path = os.path.join(output_folder, f'oscillation_data_{video_name}.csv')
    csv_result_path = os.path.join(output_folder, f'result_data_{video_name}.csv')
    plot_file_path = os.path.join(output_folder, f'oscillation_plot_{video_name}.png')
    combined_plot_file_path = os.path.join(output_folder, f'combined_oscillation_plot_{video_name}.png')

    # Initialize reference points and flag
    reference_points = {}
    first_frame_processed = False
    pattern_size_cm_id0 = 7
    pattern_size_cm = 10
    converter_value = {}

    frame_number = 0

    # Open CSV file for writing
    with open(csv_file_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # CSV Header
        csv_writer.writerow(['time', 'x_id_0', 'y_id_0', 'x_ref_id_0', 'y_ref_id_0', 'delta_x_id_0', 'delta_y_id_0',
                             'x_id_1', 'y_id_1', 'x_ref_id_1', 'y_ref_id_1', 'delta_x_id_1', 'delta_y_id_1',
                             'x_id_2', 'y_id_2', 'x_ref_id_2', 'y_ref_id_2', 'delta_x_id_2', 'delta_y_id_2'])

        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"End of video or failed to grab frame for {video_name}")
                break

            frame_number += 1

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
                    if marker_id == 0:
                        converter = float(pattern_size_cm_id0/int(height_pixel))
                    else:
                        converter = float(pattern_size_cm/int(height_pixel))
                    converter_value[marker_id[0]] = converter
                first_frame_processed = True
            
            # Special check for frame 30 with exactly 2 markers detected
            if frame_number == 30 and not first_frame_processed and ids is not None and len(ids) == 2:
                print(f"Frame {frame_number}: Only 2 markers detected")
                for i, marker_id in enumerate(ids):
                    c = corners[i][0]  # Get corners of the marker
                    reference_points[marker_id[0]] = c[0]  # Store first corner as reference point
                    height_pixel = np.linalg.norm(c[i] - c[3])
                    if marker_id == 0:
                        converter = float(pattern_size_cm_id0/int(height_pixel))
                    else:
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
                        if marker_id == 0 and frec_name == '1.5HZ':
                            ref_x, ref_y = reference_points.get(marker_id[0])
                            if ref_x is not None and ref_y is not None:
                                # Update the global variables for 1.5HZ frequency and marker_id 0
                                ref_x_id0 = ref_x
                                ref_y_id0 = ref_y
                            else:
                                print("Reference point missing for marker_id 0 at 1.5HZ")
                        else:
                            if marker_id == 0:
                                if ref_x_id0 is not None and ref_y_id0 is not None:
                                    # Use the global variables if marker_id is 0 but not 1.5HZ
                                    ref_x = ref_x_id0
                                    ref_y = ref_y_id0
                                else:
                                    print("Reference point for marker_id 0 is not set")
                                    ref_x, ref_y = 0, 0  # Provide a default or handle error
                            else:
                                # Use the reference points for other marker_ids
                                ref_x, ref_y = reference_points.get(marker_id[0])

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
                else:
                    csv_writer.writerow(current_data)

            # Display the markers on the frame
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            cv2.imshow('ArUco Marker Tracking', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    # Plotting data
    print(f"Plotting data for {video_name}...")

    # Load CSV data
    column_names = [
        'time', 'x_id_0', 'y_id_0', 'x_ref_id_0', 'y_ref_id_0', 'delta_x_id_0', 'delta_y_id_0',
        'x_id_1', 'y_id_1', 'x_ref_id_1', 'y_ref_id_1', 'delta_x_id_1', 'delta_y_id_1',
        'x_id_2', 'y_id_2', 'x_ref_id_2', 'y_ref_id_2', 'delta_x_id_2', 'delta_y_id_2'
    ]

    data = pd.read_csv(csv_file_path, header=None, names=column_names, on_bad_lines='skip')

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

    # Plot each delta_y
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
    plt.savefig(plot_file_path)

    # ---- Combined plot with max and min values ----
    plt.figure(figsize=(15, 10))

    abs_min = abs(min_delta_y_id_0)

    if abs_min > max_delta_y_id_0:
        max_y_0 = abs_min
        y_1 = abs(delta_y_id_1_at_min)
        y_2 = abs(delta_y_id_2_at_min)
        timeData = min_time
    else:
        max_y_0 = max_delta_y_id_0
        y_1 = abs(delta_y_id_1_at_max)
        y_2 = abs(delta_y_id_2_at_max)
        timeData = max_time

    
    # Plot for ID 0
    plt.plot(data['time'], data['delta_y_id_0'], label='Delta Y ID 0', marker='o', linestyle='--')

    if abs(1 - y_1) < abs(1 - y_2):
        y_meja = y_1
        # Plot for ID 1
        plt.plot(data['time'], data['delta_y_id_1'], label='Delta Y MEJA', marker='^', linestyle='--')

        # Add text annotations for the max values
        if delta_y_id_1_at_max is not None and delta_y_id_2_at_max is not None:
            plt.text(max_time, max_delta_y_id_0,
                    f'Max Delta Y Bangunan: {max_delta_y_id_0:.2f}cm\n'
                    f'Delta Y MEJA: {delta_y_id_1_at_max:.2f}cm\n',
                    color='black', fontsize=9, verticalalignment='bottom')

        # Add text annotations for the min values
        if delta_y_id_1_at_min is not None and delta_y_id_2_at_min is not None:
            plt.text(min_time, min_delta_y_id_0,
                    f'Min Delta Y Bangunan: {min_delta_y_id_0:.2f}cm\n'
                    f'Delta Y MEJA: {delta_y_id_1_at_max:.2f}cm\n',
                    color='black', fontsize=9, verticalalignment='bottom')
        
    else:
        y_meja = y_2
        # Plot for ID 2
        plt.plot(data['time'], data['delta_y_id_2'], label='Delta Y MEJA', marker='^', linestyle='--')

        # Add text annotations for the max values
        if delta_y_id_1_at_max is not None and delta_y_id_2_at_max is not None:
            plt.text(max_time, max_delta_y_id_0,
                    f'Max Delta Y Bangunan: {max_delta_y_id_0:.2f}cm\n'
                    f'Delta Y MEJA: {delta_y_id_2_at_max:.2f}cm\n',
                    color='black', fontsize=9, verticalalignment='bottom')

        # Add text annotations for the min values
        if delta_y_id_1_at_min is not None and delta_y_id_2_at_min is not None:
            plt.text(min_time, min_delta_y_id_0,
                    f'Min Delta Y Bangunan: {min_delta_y_id_0:.2f}cm\n'
                    f'Delta Y MEJA: {delta_y_id_2_at_max:.2f}cm\n',
                    color='black', fontsize=9, verticalalignment='bottom')

    # Final plot settings
    plt.legend()
    plt.title(f'Combined Oscillation of Markers\n'
              f'Max Delta Y Bangunan: {max_y_0:.2f}cm at time {timeData:.2f}seconds,\n'
              f'Delta Y MEJA: {y_meja:.2f} cm')
    
    
    # # Final plot settings with title including both max and min data
    # plt.legend()
    # plt.title(f'Combined Oscillation of Markers\n'
    #         f'Max Delta Y ID 0: {max_delta_y_id_0:.2f}cm at time {max_time:.2f}s, '
    #         f'Delta Y ID 1: {delta_y_id_1_at_max:.2f}cm, Delta Y ID 2: {delta_y_id_2_at_max:.2f}cm\n'
    #         f'Min Delta Y ID 0: {min_delta_y_id_0:.2f}cm at time {min_time:.2f}s, '
    #         f'Delta Y ID 1: {delta_y_id_1_at_min:.2f}cm, Delta Y ID 2: {delta_y_id_2_at_min:.2f}cm')
    plt.xlabel('Time (s)')
    plt.ylabel('Oscillation (cm)')
    plt.savefig(combined_plot_file_path)

    # Open CSV file for writing
    with open(csv_result_path, mode='w', newline='') as result_csv_file:
        result_csv_writer = csv.writer(result_csv_file)
        # CSV Header
        result_csv_writer.writerow(['time', 'Simpangan Maximum', 'Simpangan Meja'])
        result_csv_writer.writerow([timeData, max_y_0, y_meja])

# Loop through all videos and formats
for video in (videos_path):
    process_video(video)
