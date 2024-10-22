import os
import re

import cv2
from flask import Flask, Response, render_template

from helpers import web_helper

# Initialize the Flask app with specified template and static folder paths
app = Flask(__name__, template_folder='templates', static_folder='static')

# List to store the overlay images (header images)
overlay_image = []

# Directory containing the header images
header_img = "header_images"

# List all images in the header image directory
header_img_list = os.listdir(header_img)

# Load all header images into the overlay_image list
for i in header_img_list:
    image = cv2.imread(f'{header_img}/{i}')  # Read each image file
    overlay_image.append(image)  # Append the image to the list

@app.route('/')
def index():
    # Define the path to the 'SampleImages' directory inside the static folder
    image_folder = os.path.join(app.static_folder, 'sample_images')

    # Get all image files in the 'SampleImages' folder
    image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

    # Issue: Image files were sorted incorrectly (e.g., 1.png -> 10.png -> 2.png).
    # Reason: Character-by-character comparison, where '10' comes before '2' ('1' < '2').
    # Solution: Custom sorting functions are used to sort files numerically.

    # Sort image files numerically based on the numbers in the file names (e.g., "1.png", "2.png")
    def sort_key(filename):
        # Extract the number from the file name (before the .png extension)
        return int(filename.split('.')[0])

    # Advanced sorting function for more complex filenames (e.g., "image10.png")
    def adv_sort_key(filename):
        # Use regular expressions to find numbers in the file name
        numbers = re.findall(r'\d+', filename)
        return int(numbers[0]) if numbers else 0  # Sort by the first number found

    # Sort the image files using the simple numeric sort function
    image_files.sort(key=sort_key)

    # Prepend the static folder path to each image file for rendering in the template
    image_files = [f'/static/sample_images/{file}' for file in image_files]

    # Pass the sorted image file names to the index.html template for rendering
    return render_template('index.html', image_files=image_files)

# Generator function to continuously stream the video feed
def gen():
    cam = web_helper.VideoCamera(overlay_image=overlay_image)  # Initialize the camera with the overlay images
    t_prev = 0  # Initialize the previous timestamp for calculating FPS
    while True:
        frame, t_prev = cam.get_frame(overlay_image=overlay_image, t_prev=t_prev)  # Get the current video frame
        yield (b'--frame\r\n'  # Stream the frame data in the appropriate format
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# Route for streaming the video feed
@app.route('/video_feed')
def video_feed():
    # Return the video feed response with the correct mime type for streaming
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Run the Flask application on the local host, with debugging disabled
    app.run(host='0.0.0.0', debug=False)
    #192.168.0.105  # Commented out IP address, could be used for debugging purposes
