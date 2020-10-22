import cv2
import imutils
import datetime

def show_image(image, image_name):
    image = imutils.resize(image, width=max(800, image.shape[1]), inter=cv2.INTER_CUBIC)
    cv2.imshow(image_name, image)
    cv2.waitKey(1)
    #cv2.destroyAllWindows()

def show_processed_frame(frame, threshold, delta, text):
    # show the frame and record if the user presses a key
    frame = imutils.resize(frame, width=max(800, frame.shape[1]), inter=cv2.INTER_CUBIC)
    cv2.imshow("Security Feed", frame)
    #cv2.imshow("Gray", threshold)
    #cv2.imshow("Frame Delta", delta)
    key = cv2.waitKey(1) & 0xFF
