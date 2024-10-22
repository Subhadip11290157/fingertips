import os  # Import the os module for interacting with the operating system

import cv2  # Import OpenCV library for image processing tasks
import mediapipe as mp  # Import Mediapipe for hand tracking and gesture recognition
import numpy as np  # Import NumPy for array and numerical operations

# Define the modules and their version attributes
modules = {
    "numpy": "np.__version__",  # NumPy version
    "opencv-python": "cv2.__version__",  # OpenCV version
    "mediapipe": "mp.__version__"  # Mediapipe version
}

# Get the current working directory
current_directory = os.getcwd()

# Define the path to the requirements.txt file in the current directory
requirements_path = os.path.join(current_directory, 'requirements.txt')

# Open the requirements.txt file in write mode
# This will overwrite any existing content in the file
with open(requirements_path, 'w') as file:
    # Iterate over the modules dictionary and write their names and versions to the file
    for module_name, version_attr in modules.items():
        # Use eval() to get the version of the module
        version = eval(version_attr)
        # Write the module name and its version in the format: module_name==version
        file.write(f"{module_name}=={version}\n")

# Print a message indicating that the requirements.txt file has been successfully updated
print(f"{requirements_path} has been updated with the selected module versions.")
