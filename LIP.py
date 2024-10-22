#
# LIP == Local Independent Painter
#
import os  # Used for handling file operations
import time  # Time module to calculate FPS (Frames Per Second)

import cv2  # OpenCV library for computer vision tasks
import numpy as np  # NumPy for handling arrays and matrix operations
import pyautogui  # PyAutoGUI to get screen dimensions and perform automated GUI tasks

import helpers.track_hands as TH  # Custom helper module for hand tracking

# Get screen dimensions using PyAutoGUI
screen_width, screen_height = pyautogui.size()

# Variables for brush and eraser settings
brush_thickness = 15  # Brush thickness
eraser_thickness = 100  # Eraser thickness

# Time variables for calculating FPS
currentT = 0
previousT = 0

# Path to the directory containing overlay/header images
header_img = "header_images"
header_img_list = os.listdir(header_img)  # Get the list of images in the header folder
overlay_image = []  # List to store loaded overlay images

# Load all header images into overlay_image list
for i in header_img_list:
    image = cv2.imread(f"{header_img}/{i}")  # Read each image file
    overlay_image.append(image)  # Add to list of overlays

# Open webcam feed using OpenCV
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Set webcam properties: width, height, and FPS
cap.set(3, screen_width)  # Width
cap.set(4, screen_height)  # Height
cap.set(cv2.CAP_PROP_FPS, 60)  # Frames per second

# Set the default overlay image and brush color
default_overlay = overlay_image[0]  # Initial header image
draw_color = (81, 242, 56)  # Default color (leafy green)

# Initialize hand tracking using the custom helper module
detector = TH.handDetector(min_detection_confidence=0.85)

# Variables to store previous coordinates (initially zero)
xp = 0
yp = 0

# Create a blank canvas (black background) matching the screen size for drawing
image_canvas = np.zeros((screen_height, screen_width, 3), np.uint8)

# Main loop for real-time hand tracking and drawing
running = True
while running:
    ret, frame = cap.read()  # Capture frame from webcam
    
    if not ret:
        break  # Exit loop if the webcam feed is not available

    # Flip the frame horizontally to create a mirror effect
    frame = cv2.flip(frame, 1)

    # Get the current frame's dimensions after flipping
    frame_height, frame_width, _ = frame.shape
    
    # Resize the frame to match screen dimensions (upscaling)
    frame = cv2.resize(frame, (screen_width, screen_height))

    # Resize the default overlay image to match the screen width and apply it to the frame
    default_overlay_resized = cv2.resize(default_overlay, (screen_width, 125))
    frame[0:125, 0:screen_width] = default_overlay_resized

    # Detect hands in the frame and get the list of landmarks (finger positions)
    frame = detector.findHands(frame, draw=True)
    landmark_list = detector.findPosition(frame, draw=False)

    # If landmarks are detected, proceed with gesture recognition
    if len(landmark_list) != 0:
        x1, y1 = landmark_list[8][1:]  # Index finger tip position
        x2, y2 = landmark_list[12][1:]  # Middle finger tip position

        # Get the status of each finger (up/down)
        my_fingers = detector.fingerStatus()

        # Selection mode: both index and middle fingers are up
        if my_fingers[1] and my_fingers[2]:
            # Check if the hand is over the toolbar (header)
            if y1 < 125:
                # Change colors and overlays based on finger position within the header
                if 355 < x1 < 460:
                    default_overlay = overlay_image[0]
                    draw_color = (255, 255, 0)  # Aqua blue
                elif 475 < x1 < 560:
                    default_overlay = overlay_image[1]
                    draw_color = (47, 225, 245)  # Yellow
                elif 610 < x1 < 685:
                    default_overlay = overlay_image[2]
                    draw_color = (197, 47, 245)  # Pink
                elif 755 < x1 < 865:
                    default_overlay = overlay_image[3]
                    draw_color = (81, 242, 56)  # Leafy green
                elif 1060 < x1 < 1220:
                    default_overlay = overlay_image[4]
                    draw_color = (0, 0, 0)  # Black (eraser mode)

            # Display "SELECT Mode" on the screen
            cv2.putText(
                frame,
                "SELECT Mode",
                (900, 680),
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                color=(0, 255, 255),
                thickness=2,
                fontScale=0.9,
            )
            # Draw a line between the tips of the index and middle fingers
            cv2.line(frame, (x1, y1), (x2, y2), color=draw_color, thickness=3)

        # Paint mode: only index finger is up
        if my_fingers[1] and not my_fingers[2]:
            # Display "PAINT Mode" on the screen
            cv2.putText(
                frame,
                "PAINT Mode",
                (900, 680),
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                color=(0, 255, 255),
                thickness=2,
                fontScale=0.9,
            )
            # Draw a circle at the tip of the index finger
            cv2.circle(frame, (x1, y1), 15, draw_color, thickness=-1)

            # Logic for drawing lines: use previous (xp, yp) and current (x1, y1) finger coordinates
            if xp == 0 and yp == 0:  # Initial point check
                xp, yp = x1, y1

            # Eraser mode: draw thicker black lines
            if draw_color == (0, 0, 0):
                cv2.line(frame, (xp, yp), (x1, y1), color=draw_color, thickness=eraser_thickness)
                cv2.line(image_canvas, (xp, yp), (x1, y1), color=draw_color, thickness=eraser_thickness)
            else:  # Paint mode: draw colored lines
                cv2.line(frame, (xp, yp), (x1, y1), color=draw_color, thickness=brush_thickness)
                cv2.line(image_canvas, (xp, yp), (x1, y1), color=draw_color, thickness=brush_thickness)

            # Update the previous point to the current point
            xp, yp = x1, y1  # (i)

    # Convert the drawing canvas to grayscale
    img_gray = cv2.cvtColor(image_canvas, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to create a binary inverse image
    _, imginv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
    imginv = cv2.cvtColor(imginv, cv2.COLOR_GRAY2BGR)  # Convert back to 3-channel color

    # Resize the inverted image to match the screen dimensions
    imginv = cv2.resize(imginv, (screen_width, screen_height))

    # Combine the frame and the inverse binary image using bitwise operations (refer README for detailed logic)
    frame2 = cv2.bitwise_and(frame, imginv) # (iv) -> (a), (b) in the "NOTES" section below
    frame = cv2.bitwise_or(frame2, image_canvas) # (iv) -> (c), (d) in the "NOTES" section below

    # Calculate FPS (frames per second)
    currentT = time.time()
    fps = 1 / (currentT - previousT)
    previousT = currentT

    # Display FPS on the screen
    cv2.putText(
        frame,
        "Render FPS:" + str(int(fps)),
        (10, 685),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.8,
        color=(0, 0, 255),
        thickness=2,
    )

    # Display the final frame with drawings and overlays
    cv2.imshow("Play_with_Paint", frame)

    # Wait for 1 ms and check if 'Esc' or 'q' is pressed to exit the loop
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord("q"):  # 'Esc' key or 'q' to quit
        break

    # Check if the window was closed (using getWindowProperty)
    # WARNING: Here, use cv2.getWindowProperty(...) only after cv2.waitkey(...) has been called (v)
    if cv2.getWindowProperty("Play_with_Paint", cv2.WND_PROP_VISIBLE) < 1:
        running = False
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

# NOTES:

# (i): Before drawing the next frame, update the coordinates to the latest finger position.

# (ii): Thresholding step ensures that pixels with intensity <= 50 are turned white, and those > 50 are turned black (inversion).

# (iii): Convert: just to make it 3d np array from 1d np array for compatibility while overlapping with 3d images (np arrays) ahead in steps where 
    # bitwise operations are done between images (3d and 1d images aren't compatible for bitwise operations, images being compared and operated 
    # on bit-by-bit must have same dimension for unambiguous execution)

# (iv) logics used:

    # (a): 1 AND X == X, as per this Boolean rule, white_bg_of_bin_inv_img AND the_frame_bg will gives the frame_bg because white is equivalent to 1

    # (b): 0 AND X == 0, as per this Boolean rule, black_lines_on_bin_inv_img AND colored_lines_on_frame gives the black lines on frame because black is equivalent to 0

    # (c): 1 OR X == 1, as per this Boolean rule, colorful_lines_on_image_canvas OR black_lines_on_frame2 gives colorful lines on frame2 because colorful is non-zero i.e. non-false i.e. TRUE, black is FALSE and TRUE OR FALSE gives TRUE

    # (d): 0 OR X == X, as per this Boolean rule, black_bg_of_img_canvas OR non_black_bg_of_frame2 gives the latter one because of similar comparisons of colors with TRUE and FALSE values as done above.


# (v) Why? 

   # Answer: because waitkey not only checks for key-presses but also any kind of "window events" (which includes window close events)
   # so when you close window by clicking  the 'X' button, the function-call: cv2.waitkey() registers it as an "event"
   # now you can check for "cv2.getWindowProperty" and it will give you a value < 1 i.e. window is NOT open. And so you break out of the loop. As simple as that.

   # Why cv2.waitkey() is not enough to break out of the loop?

   # Answer: you closed the window using a mouse-event which is not a key-press hence the above if condition will not be true and you will NOT be able to break out 
   # of the loop. So we need the below function-call to verify the window visible state (visible if open i.e. value == 1, else closed) is not open i.e. < 
   # 1 and thus break out from the while loop. This function doesn't rely on key-press, rather directly checks the window's open/close status.

   # What if I use cv2.getWindowProperty() before cv2.waitKey() ?

   # Answer: You will see another window opening everytime you attempt to close the current window i.e. you won't break out of the while loop ever
   # Reason: python interpreter follows top-down scanning of code. When you click the close button, the window actually closes. Its true. 
   # BUT -> cv2 doesn't know it has closed because to register such event in the event loop of opencv, you need to call the cv2.waitkey() function 
   # so at the present moment, cv2 thinks the window is STILL OPEN! and so  cv2.getWindowProperty(...) will return 1 (i.e. OPEN) and again, 
   # you can't break out of the while loop.
   
   
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------