import cv2
import numpy as np
import imutils.object_detection
import Core.Inputs
from Core.Viewer import show_image, show_processed_frame

def find_motion_from_stream(stream):
    firstFrame = None
    while True:
        # grab the current frame and initialize the occupied/unoccupied
        # text
        frame = stream.read()
        text = "Unoccupied"
        # if the frame could not be grabbed, then we have reached the end
        # of the video
        if frame is None or not frame[0]:
            break
        #frame = shadow_remove(frame[1])
        frame = frame[1]
        # resize the frame, convert it to grayscale, and blur it
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (31, 31), 0)
        # if the first frame is None, initialize it
        if firstFrame is None:
            firstFrame = gray
            continue

        # compute the absolute difference between the current frame and
        # first frame
        frameDelta = cv2.absdiff(firstFrame, gray)
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
        # dilate the thresholded image to fill in holes, then find contours
        # on thresholded image
        thresh = cv2.dilate(thresh, None, iterations=2)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        # loop over the contours
        for i,c in enumerate(cnts):
            # if the contour is too small, ignore it
            if cv2.contourArea(c) < 200:
                continue
            # compute the bounding box for the contour, draw it on the frame,
            # and update the text
            (x, y, w, h) = cv2.boundingRect(c)
            frame_slice = frame[max(0,y-h):min(y+h,frame.shape[0]), max(0,x-w):min(x+w,frame.shape[1])]
            #if contains_pedestrian(frame):
                #show_image(frame_slice, f"image")
            #    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            #elif contains_car(frame):
             #   cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            #else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = "Occupied"

        #Apply Pedestrian Detection
        #frame = find_pedestrians_from_image(frame)

        show_processed_frame(frame, gray, frameDelta, text)

    stream.stop()
    cv2.destroyAllWindows()


def contains_car(image):
    return contains_feature(image, "cars.xml")


def contains_pedestrian(image):
    image = imutils.resize(image, width=min(400, image.shape[1]))
    hog_classifier = cv2.HOGDescriptor()
    hog_classifier.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    (rects, weights) = hog_classifier.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = imutils.object_detection.non_max_suppression(rects, probs=None, overlapThresh=0.65)
    return len(pick) > 0


def find_pedestrians_from_image(image):
    image = imutils.resize(image, width=min(400, image.shape[1]))
    hog_classifier = cv2.HOGDescriptor()
    hog_classifier.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    (rects, weights) = hog_classifier.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = imutils.object_detection.non_max_suppression(rects, probs=None, overlapThresh=0.65)
    # draw the final bounding boxes
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(image, (xA, yA), (xB, yB), (255, 0, 0), 2)
    return image

def shadow_remove(img):
    rgb_planes = cv2.split(img)
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((10,10), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_norm_planes.append(norm_img)
    shadowremov = cv2.merge(result_norm_planes)
    return shadowremov
#Shadow removal

def contains_feature(image, classifier_name = "haarcascade_fullbody.xml"):
    # Convert to grey
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    classifier = cv2.CascadeClassifier(cv2.data.haarcascades + classifier_name)
    objects = classifier.detectMultiScale(image_gray, 1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    return len(objects) > 0




def translate_image(image, row_shift, col_shift):
    rows, cols, _ = image.shape
    M = np.float32([[1, 0, row_shift], [0, 1, col_shift]])
    dst = cv2.warpAffine(image, M, (cols, rows))
    return dst

def erode_image(image, kernel_size):
    return cv2.erode(image, np.ones(kernel_size, np.uint8))

def dilate_image(image, kernel_size):
    return cv2.dilate(image, np.ones(kernel_size, np.uint8))

def open_image(image, kernel_size):
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, np.ones(kernel_size, np.uint8))

def find_image_gradient(image, kernel_size):
    return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, np.ones(kernel_size, np.uint8))
