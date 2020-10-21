import requests
import json
import cv2
import numpy as np
import datetime
import time



with open('sensitive_info.json') as info_file:
    info = json.load(info_file)
    ip_address = info.get("ip")
    username = info.get('username')
    password = info.get('password')
    channels = set(info.get('channels'))


def capture_image_from_camera(channel):
    save_image(get_new_cv2_camera_screenshot(channel))
    print("Success!!")


def save_image(image):
    timestamp = str(datetime.datetime.now())
    filename = f"{timestamp}.jpg"
    f = open(filename, 'x')
    f.close()
    cv2.imwrite(filename, image)


def get_new_cv2_camera_screenshot(channel):
    image = convert_to_cv_object(make_screen_shot(channel))
    image = cv2.resize(image, None,fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC)
    return image

def load_cv2_from_saved(path, resize_ratios = (1,1)):
    image = cv2.imread(path)
    if resize_ratios != (1,1):
        image = cv2.resize(image, None,fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC)
    return image

def make_screen_shot(channel):
    if channel not in channels:
        raise Exception("Channel not found")
    call_string = f"http://{ip_address}/cgi-bin/api.cgi?cmd=Snap&channel={channel}&rs=wuuPhkmUCeI9WG7C&user={username}&password={password}"
    reolink_response = requests.get(call_string, stream = True)
    if reolink_response.status_code != 200:
        raise Exception("Request Failed!")
    return reolink_response


def convert_to_matrix(http_response):
    return np.asarray(bytearray(http_response.content), dtype = "uint8")

def convert_to_cv_object(http_response):
    return cv2.imdecode(http_response.content, cv2.IMREAD_COLOR)





def main():
    pass


if __name__== '__main__':
    main()
