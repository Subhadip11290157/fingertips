import os

import cv2
import mediapipe as mp
# Import the required modules
import numpy as np

# Define the modules and their version attributes
modules = {
    "numpy": "np.__version__",
    "opencv-python": "cv2.__version__",
    "mediapipe": "mp.__version__"
}

# Get the current working directory and find the path to the requirements.txt file
current_directory = os.getcwd()
requirements_path = os.path.join(current_directory, 'requirements.txt')

# Open the requirements.txt file in write mode to overwrite the existing content
with open(requirements_path, 'w') as file:
    # Iterate over the modules and write their versions to the file
    for module_name, version_attr in modules.items():
        version = eval(version_attr)
        file.write(f"{module_name}=={version}\n")

print(f"{requirements_path} has been updated with the selected module versions.")
