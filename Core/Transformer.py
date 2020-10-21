import cv2
import numpy as np
import Core.Inputs
from Core.Viewer import show_image

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


def find_feature(image, classifier=face_cascade):
    # Convert to grey
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(image_gray, 1.3, 5)
    for (x, y, w, h) in faces:
        img = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = image_gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        show_image(image)






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
