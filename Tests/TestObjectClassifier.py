import cv2
import imutil
import Core.Inputs as Inputs
import Core.Transformer as Transformer
import Core.Viewer as Viewer
from os.path import realpath, normpath




def test_human_recogniton_from_save(path, classifier_name = "haarcascade_frontalface_default.xml"):
    image = Inputs.load_cv2_from_saved(path, (0.3,0.3))
    Transformer.find_feature(image, classifier_name)

def test_pedestrian_recognition_from_save(path):
    image = Inputs.load_cv2_from_saved(path, (0.3,0.3))
    Transformer.find_pedestrians_from_image(image)

def test_pedestrian_recognition_from_camera(channel):
    image = Inputs.load_cv2_from_camera(channel)
    Transformer.find_pedestrians_from_image(image)

def test_motion_detection_from_camera(channel):
    stream = Inputs.open_video_stream(channel)
    Transformer.find_motion_from_stream(stream)









def main():
    #test_pedestrian_recognition_from_camera(3)
    test_motion_detection_from_camera(3)
    #test_pedestrian_recognition_from_save("2020-10-21 00:51:14.271985.jpg")


if __name__ == '__main__':
    main()

