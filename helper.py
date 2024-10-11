import os

import cv2
import numpy as np

import track_hands as TH  # Importing hand tracking module


class VideoCamera():
    def __init__(self, overlay_image=[], draw_color=(255, 200, 100)):
        # Initialize video capture (webcam)
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(3, 1280)  # Set the frame width
        self.cap.set(4, 720)   # Set the frame height
        self.xp = 0  # Previous x-coordinate for drawing
        self.yp = 0  # Previous y-coordinate for drawing
        self.x1 = 0  # Current x-coordinate for index finger
        self.y1 = 0  # Current y-coordinate for index finger
        self.x2 = 0  # Current x-coordinate for middle finger
        self.y2 = 0  # Current y-coordinate for middle finger
        self.brush_thickness = 15  # Thickness of the brush for drawing
        self.eraser_thickness = 100  # Thickness of the eraser
        self.overlay_image = overlay_image  # List of images to overlay (color palette)
        self.draw_color = draw_color  # Initial drawing color (RGB)
        
        # Initialize hand detection using track_hands module with a minimum confidence level
        self.detector = TH.handDetector(min_detection_confidence=0.85)
        
        # A blank canvas where drawing will be stored
        self.image_canvas = np.zeros((720, 1280, 3), np.uint8)
        
        # Set the default overlay image (first one from the list)
        self.default_overlay = overlay_image[0]

    def __del__(self):
        # Release the webcam when the object is destroyed
        self.cap.release()

    def set_overlay(self, frame, overlay_image):
        # Set the default overlay (header) image
        self.default_overlay = overlay_image[0]
        frame[0:125, 0:1280] = self.default_overlay  # Display the overlay image on top of the frame
        return frame

    def get_frame(self, overlay_image):
        # Capture the current frame from the webcam
        _, frame = self.cap.read()
        frame = cv2.flip(frame, 1)  # Flip the frame horizontally (mirror effect)
        
        # Set the header image (palette) at the top
        frame[0:125, 0:1280] = self.default_overlay
        
        # Detect hands and find hand landmarks (without drawing landmarks on the frame)
        frame = self.detector.findHands(frame, draw=True)
        landmark_list = self.detector.findPosition(frame, draw=False)

        if len(landmark_list) != 0:
            # Get the positions of the index finger (landmark 8) and middle finger (landmark 12)
            self.x1, self.y1 = (landmark_list[8][1:])  # Index finger tip
            self.x2, self.y2 = landmark_list[12][1:]  # Middle finger tip

            # Get the status of fingers (which are up or down)
            my_fingers = self.detector.fingerStatus()

            # If both index and middle fingers are up, enter selection mode (color selector)
            if my_fingers[1] and my_fingers[2]:
                # Reset previous drawing points
                self.xp, self.yp = 0, 0

                # If the hand is within the top color palette area, change the color
                if self.y1 < 125:
                    # Change color based on the x-position of the index finger
                    if 200 < self.x1 < 340:
                        self.default_overlay = overlay_image[0]
                        frame[0:125, 0:1280] = self.default_overlay
                        self.draw_color = (255, 0, 0)  # Blue
                    elif 340 < self.x1 < 500:
                        self.default_overlay = overlay_image[1]
                        self.draw_color = (47, 225, 245)  # Cyan
                        frame[0:125, 0:1280] = self.default_overlay
                    elif 500 < self.x1 < 640:
                        self.default_overlay = overlay_image[2]
                        self.draw_color = (197, 47, 245)  # Purple
                        frame[0:125, 0:1280] = self.default_overlay
                    elif 640 < self.x1 < 780:
                        self.default_overlay = overlay_image[3]
                        self.draw_color = (53, 245, 47)  # Green
                        frame[0:125, 0:1280] = self.default_overlay
                    elif 1100 < self.x1 < 1280:
                        self.default_overlay = overlay_image[4]
                        self.draw_color = (0, 0, 0)  # Eraser (Black)
                        frame[0:125, 0:1280] = self.default_overlay

                # Display text for color selector mode
                cv2.putText(frame, 'Color Selector Mode', (900, 680), fontFace=cv2.FONT_HERSHEY_COMPLEX, color=(0, 255, 255), thickness=2, fontScale=1)

                # Draw a line connecting index and middle fingers
                cv2.line(frame, (self.x1, self.y1), (self.x2, self.y2), color=self.draw_color, thickness=3)

            # If only the index finger is up, enter writing mode (drawing on canvas)
            if my_fingers[1] and not my_fingers[2]:
                cv2.putText(frame, "Writing Mode", (900, 680), fontFace=cv2.FONT_HERSHEY_COMPLEX, color=(255, 255, 0), thickness=2, fontScale=1)
                
                # Draw a circle at the tip of the index finger (brush tip)
                cv2.circle(frame, (self.x1, self.y1), 15, self.draw_color, thickness=-1)

                # Reset previous points if not initialized
                if self.xp == 0 and self.yp == 0:
                    self.xp = self.x1
                    self.yp = self.y1

                # If using the eraser (draw_color is black), draw thicker lines
                if self.draw_color == (0, 0, 0):
                    cv2.line(frame, (self.xp, self.yp), (self.x1, self.y1), color=self.draw_color, thickness=self.eraser_thickness)
                    cv2.line(self.image_canvas, (self.xp, self.yp), (self.x1, self.y1), color=self.draw_color, thickness=self.eraser_thickness)
                else:
                    # Draw lines with the brush color and thickness
                    cv2.line(frame, (self.xp, self.yp), (self.x1, self.y1), color=self.draw_color, thickness=self.brush_thickness)
                    cv2.line(self.image_canvas, (self.xp, self.yp), (self.x1, self.y1), color=self.draw_color, thickness=self.brush_thickness)

                # Update previous points to current points
                self.xp, self.yp = self.x1, self.y1

        # Add the header image (color palette) at the top
        frame[0:125, 0:1280] = self.default_overlay

        # Convert the drawing canvas to grayscale and create an inverse mask
        img_gray = cv2.cvtColor(self.image_canvas, cv2.COLOR_BGR2GRAY)
        _, imginv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
        imginv = cv2.cvtColor(imginv, cv2.COLOR_GRAY2BGR)

        # Combine the frame with the drawing canvas using bitwise operations
        frame = cv2.bitwise_and(frame, imginv)  # Mask the frame where drawing exists
        frame = cv2.bitwise_or(frame, self.image_canvas)  # Add the drawing on top of the frame

        # Encode the final frame as JPEG to be sent for rendering
        _, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

def main():
    overlay_image = []  # List to store header images (color palettes)
    header_img = "header_images"
    
    # Load all header images from the directory
    header_img_list = os.listdir(header_img)
    for i in header_img_list:
        image = cv2.imread(f'{header_img}/{i}')
        overlay_image.append(image)

    # Initialize the VideoCamera object with the overlay images
    cam1 = VideoCamera(overlay_image=overlay_image)

    while True:
        # Capture frame from the webcam
        ret, input_img = cam1.cap.read()
        input_img = cv2.flip(input_img, 1)  # Flip the frame horizontally
        
        # Process the frame and overlay images
        my_frame = cam1.get_frame(frame=input_img, overlay_image=overlay_image)

        # Display the output frame
        cv2.imshow('out', my_frame)
        cv2.waitKey(1)  # Wait for 1 millisecond before processing the next frame

if __name__ == "__main__":
    main()
