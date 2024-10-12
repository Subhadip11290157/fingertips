import os
import re

import cv2
from flask import Flask, Response, render_template

from helpers import web_helper

app = Flask(__name__,template_folder='templates', static_folder='static')
overlay_image=[]
header_img = "header_images"
header_img_list = os.listdir(header_img)

for i in header_img_list:
    image = cv2.imread(f'{header_img}/{i}')
    overlay_image.append(image)

@app.route('/')
def index():
    # Get the list of images from the 'SampleImages' directory
    image_folder = os.path.join(app.static_folder, 'sample_images')
    image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

    # an issue occurred:
    # 1.png -> 10.png -> 11.png ....-> 15.png -> 2.png - 3.png .....-> 9.png
    # this was the order being taken, without customized sorting
    # reason: acc. to char-by-char comparison, "10.png" < "2.png" as '1' < '2'
    # to solve this issue, these custom sorters are designed :-

    # Sort the image files numerically based on the numbers in the file names
    def sort_key(filename):
        # Extract the number from the file name (before .png)
        return int(filename.split('.')[0])

    # Customized advanced sorting for complex filenames like "image1.png"
    def adv_sort_key(filename):
        # Use regex to extract numbers from filenames like "image10.png"
        numbers = re.findall(r'\d+', filename)
        return int(numbers[0]) if numbers else 0  # Sort by the first number found
    
    # Sort based on the extracted number
    image_files.sort(key=sort_key)  

    # Pass the image file names to the template
    return render_template('index.html', image_files=image_files)

def gen():
    cam = web_helper.VideoCamera(overlay_image= overlay_image)

    while True:
        frame = cam.get_frame(overlay_image=overlay_image)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)
    #192.168.0.105