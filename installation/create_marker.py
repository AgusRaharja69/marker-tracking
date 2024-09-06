import cv2

# Load the dictionary that defines the type of ArUco marker
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# Define the marker ID (choose an ID from 0 to 49 for DICT_4X4_50)
marker_id = 2  # You can change this to generate other markers

# Define the size of the marker in pixels
marker_size = 1000  # The marker will be 200x200 pixels

# Generate the marker image
marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)

# Save the marker image to a file
cv2.imwrite("simple_aruco_marker1.png", marker_img)

# Display the generated marker image
cv2.imshow("ArUco Marker", marker_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
