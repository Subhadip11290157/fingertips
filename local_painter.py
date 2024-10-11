import os
import time

import cv2
import numpy as np

import track_hands as TH

brush_thickness = 15
eraser_thickness = 100
image_canvas = np.zeros((720, 1280, 3), np.uint8)

currentT = 0
previousT = 0

header_img = "header_images"
header_img_list = os.listdir(header_img)
overlay_image = []


for i in header_img_list:
    image = cv2.imread(f"{header_img}/{i}")
    overlay_image.append(image)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 1280)
cap.set(4, 720)
cap.set(cv2.CAP_PROP_FPS, 60)

default_overlay = overlay_image[0]
draw_color = (255, 200, 100)

detector = TH.handDetector(min_detection_confidence=0.85)

xp = 0
yp = 0

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame[0:125, 0:1280] = default_overlay

    frame = detector.findHands(frame, draw=True)
    landmark_list = detector.findPosition(frame, draw=False)

    if len(landmark_list) != 0:
        x1, y1 = landmark_list[8][1:]  # index
        x2, y2 = landmark_list[12][1:]  # middle

        my_fingers = detector.fingerStatus()
        # print(my_fingers)
        if my_fingers[1] and my_fingers[2]:  # if both are up
            if y1 < 125:
                if 355 < x1 < 460:
                    default_overlay = overlay_image[0]
                    draw_color = (235, 206, 135) # sky blue
                elif 475 < x1 < 560:
                    default_overlay = overlay_image[1]
                    draw_color = (47, 225, 245) # yellow
                elif 610 < x1 < 685:
                    default_overlay = overlay_image[2]
                    draw_color = (197, 47, 245) # pink
                elif 755 < x1 < 865:
                    default_overlay = overlay_image[3]
                    draw_color = (4, 169, 92)  # leaf green
                elif 1060 < x1 < 1220:
                    default_overlay = overlay_image[4]
                    draw_color = (0, 0, 0)  # black

            cv2.putText(
                frame,
                "Color Selector Mode",
                (900, 680),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                color=(0, 255, 255),
                thickness=2,
                fontScale=1,
            )
            cv2.line(frame, (x1, y1), (x2, y2), color=draw_color, thickness=3)

        if my_fingers[1] and not my_fingers[2]:  # index == up, middle == not up

            cv2.putText(
                frame,
                "Writing Mode",
                (900, 680),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                color=(255, 255, 0),
                thickness=2,
                fontScale=1,
            )
            cv2.circle(frame, (x1, y1), 15, draw_color, thickness=-1)

            # logic: to draw a line: tell opencv line draw function the prev (x,y) and current (x,y)

            if (
                xp == 0 and yp == 0
            ):  # if not started drawing i.e. when cursor is at default == top left corner
                xp = x1
                yp = y1

            if draw_color == (0, 0, 0):  # when eraser selected
                cv2.line(
                    frame,
                    (xp, yp),
                    (x1, y1),
                    color=draw_color,
                    thickness=eraser_thickness,
                )
                cv2.line(
                    image_canvas,
                    (xp, yp),
                    (x1, y1),
                    color=draw_color,
                    thickness=eraser_thickness,
                )

            else:  # when any painbrush selected
                cv2.line(
                    frame,
                    (xp, yp),
                    (x1, y1),
                    color=draw_color,
                    thickness=brush_thickness,
                )
                cv2.line(
                    image_canvas,
                    (xp, yp),
                    (x1, y1),
                    color=draw_color,
                    thickness=brush_thickness,
                )

        xp, yp = x1, y1  # (i)

    img_gray = cv2.cvtColor(
        image_canvas, cv2.COLOR_BGR2GRAY
    )  # colorful -> black n' white (grayscale)

    # cv2.imshow("canvas image", image_canvas) # the lines drawn on black canvas
    # cv2.imshow("greyscaled image", img_gray) # varieties of gray (0 to 255) unless thresholded

    _, imginv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)  # (ii)
    imginv = cv2.cvtColor(imginv, cv2.COLOR_GRAY2BGR)  # (iii)

    # cv2.imshow("bin_inv_img", imginv) # 3d (RGB) but black n' white image

    frame2 = cv2.bitwise_and(frame, imginv)  # (a), (b) below

    # cv2.imshow("frame AND bin_inv_img",frame2) # (c), (d) below

    frame = cv2.bitwise_or(frame2, image_canvas)
    currentT = time.time()
    fps = 1 / (currentT - previousT)
    previousT = currentT

    cv2.putText(
        frame,
        "Client FPS:" + str(int(fps)),
        (10, 670),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=(255, 0, 0),
        thickness=2,
    )
    
    cv2.imshow("Play_with_Paint", frame)

    # Wait for 1 millisecond and check if 'Esc' or 'q' is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord("q"):  # 27 is the ASCII code for the 'Esc' key
        break


# -------------------------------------------------------------------------------------

# notes:
#
# (i): before next frame, co-ordinates are updated to point to the latest position of the tip of the index finger tracked in this iteration
#
# (ii): pixels with intensity <=50 gets 255 (white) and those >50 gets 0 (black), so now  the image will be pure black on pure white background (inversed + binarized)
#
# (iii): just to make it 3d np array from 1d np array for compatibility while overlapping with 3d images (np arrays) ahead in steps where bitwise operations are done between images (3d and 1d images aren't compatible for bitwise operations, images being compared and operated on bit-by-bit must have same dimension for unambiguous execution)


# logics used:
# (a): 1 AND X == X, as per this Boolean rule, white_bg_of_bin_inv_img AND the_frame_bg will gives the frame_bg because white is equivalent to 1
#
# (b): 0 AND X == 0, as per this Boolean rule, black_lines_on_bin_inv_img AND colored_lines_on_frame gives the black lines on frame because black is equivalent to 0
#
# (c): 1 OR X == 1, as per this Boolean rule, colorful_lines_on_image_canvas OR black_lines_on_frame2 gives colorful lines on frame2 because colorful is non-zero i.e. non-false i.e. TRUE, black is FALSE and TRUE OR FALSE gives TRUE
#
# (d): 0 OR X == X, as per this Boolean rule, black_bg_of_img_canvas OR non_black_bg_of_frame2 gives the latter one because of similar comparisons of colors with TRUE and FALSE values as done above.

# -------------------------------------------------------------------------------------
