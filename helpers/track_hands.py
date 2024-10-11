import time

import cv2
import mediapipe as mp


class handDetector:
    def __init__(
        self,
        image_mode=False,                   # Whether to treat input images as a batch (for static images)
        max_num_hands=3,                    # Maximum number of hands to detect
        modelComplexity=1,                  # Higher complexity model for better accuracy
        min_detection_confidence=0.5,       # Minimum confidence to detect a hand
        min_tracking_confidence=0.5,        # Minimum confidence to track hand landmarks
    ):
        self.image_mode = image_mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.modelComplex = modelComplexity

        # Initialize MediaPipe's hands solution and drawing utilities
        self.mphands = mp.solutions.hands
        self.hands = self.mphands.Hands(
            self.image_mode,
            self.max_num_hands,
            self.modelComplex,
            self.min_detection_confidence,
            self.min_tracking_confidence,
        )
        self.mpdraw = mp.solutions.drawing_utils  # For drawing hand landmarks on the image
        self.finger_tip_id = [4, 8, 12, 16, 20]  # Landmark IDs for thumb, index, middle, ring, and pinky tips

    def findHands(self, img, draw=True):
        # Convert BGR image to RGB (MediaPipe expects RGB input)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Process the image and detect hands
        self.results = self.hands.process(imgRGB)

        # If hands are detected, loop over each hand and draw landmarks
        if self.results.multi_hand_landmarks:
            for i in self.results.multi_hand_landmarks:
                if draw:
                    # Draw landmarks and connections between them on the image
                    self.mpdraw.draw_landmarks(img, i, self.mphands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, hand_num=0, draw=True):
        # List to store landmark positions of the detected hand
        self.lm_list = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[hand_num]  # Get the specific hand based on index (hand_num)
            for id, lm in enumerate(myHand.landmark):
                # Convert normalized landmarks to pixel coordinates
                h, w, c = img.shape  # Get image dimensions
                cx, cy = int(lm.x * w), int(lm.y * h)  # Convert landmark coordinates to pixel positions
                self.lm_list.append([id, cx, cy])  # Add to list: [landmark ID, x-pos, y-pos]
                
                if draw:
                    # Optionally draw circles at each landmark point
                    cv2.circle(
                        img,
                        center=(cx, cy),
                        radius=3,
                        color=(255, 255, 255),
                        thickness=1,
                    )
        return self.lm_list

    def fingerStatus(self):
        fingers = []

        # Check the thumb's state (fingers[0])
        if (
            self.lm_list[self.finger_tip_id[0]][1]  # x-coord of the thumb tip (landmark 4)
            < self.lm_list[self.finger_tip_id[0] - 1][1]  # x-coord of the thumb joint (landmark 3)
        ):
            # Thumb is open (i)
            fingers.append(1)
        else:
            fingers.append(0)

        # Check the status of the remaining four fingers (index, middle, ring, little)
        for i in range(1, 5):
            # Compare y-coordinates of the finger tip (landmarks 8, 12, 16, 20) and the joint two steps below (landmarks 6, 10, 14, 18)
            if (
                self.lm_list[self.finger_tip_id[i]][2]  # y-coord of the fingertip
                < self.lm_list[self.finger_tip_id[i] - 2][2]  # y-coord of the knuckle below the tip
            ):
                # Finger is open (ii)
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers


def main():
    cap = cv2.VideoCapture(0)  # Capture video from the default camera
    previousT = 0  # To track time for FPS calculation
    currentT = 0

    detector = handDetector()  # Initialize handDetector object

    while True:
        ret, img = cap.read()  # Read a frame from the webcam

        img = detector.findHands(img, draw=True)  # Detect hands and optionally draw landmarks
        landmark_list = detector.findPosition(img)  # Get the positions of landmarks
        if len(landmark_list) != 0:
            # If landmarks are found, print the coordinates of landmark 2 (index finger joint)
            print(landmark_list[2])

        # FPS calculation
        currentT = time.time()
        fps = 1 / (currentT - previousT)  # Calculate frames per second
        previousT = currentT

        # Display the calculated FPS on the image
        cv2.putText(
            img,
            "Client FPS:" + str(int(fps)),
            (10, 70),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=2,
            color=(255, 0, 0),
        )

        cv2.imshow("img", img)  # Show the frame with hand detection
        cv2.waitKey(1)  # Wait for 1 millisecond before moving to the next frame


if __name__ == "__main__":
    main()



# -------------------------------------------------------------------------------------

# notes:
#
# (i): if x-cord of tip of thumb (landmark point-4) is left of the x-cord of the knuckle joint (landmark point-3) just below it (this usually happens when right hand palm is opened with fingers stretched out)
#
# (ii): if y-cord of the tip of a finger (index, middle, ring or little finger) is logically higher than the y-co-ordinate of the 2nd knuckle joint below it (see when you close your fist, when the fingers are not raised, then this condition is satisfied) then it indicates the finger is down (NOT up).
# So why the '<' sign?
# Reason: in OpenCV, the origin of the coordinate system is always at the top-left corner of the image or canvas. This is a standard convention used in many image processing libraries, including OpenCV.
# Thus, the measurement of y-co-ordinate values happen upside down resulting in a lower value for a taller point and higher value for a shorter point on canvas. Hence the conditionality gets reversed.

# -------------------------------------------------------------------------------------
