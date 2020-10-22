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
    snap_channels = set(info.get('snap_channels'))
    recording_channels = set(info.get('recording_channels'))


def capture_image_from_camera(channel):
    save_image(load_cv2_from_camera(channel))
    print("Success!!")


def save_image(image):
    timestamp = str(datetime.datetime.now())
    filename = f"{timestamp}.jpg"
    f = open(filename, 'x')
    f.close()
    cv2.imwrite(filename, image)


def load_cv2_from_camera(channel, resize_ratios = (1,1)):
    image = convert_to_cv_object(make_screen_shot(channel))
    if resize_ratios != (1,1):
        image = cv2.resize(image, None,fx=resize_ratios[0], fy=resize_ratios[1], interpolation=cv2.INTER_CUBIC)
    return image

def load_cv2_from_saved(path, resize_ratios = (1,1)):
    image = cv2.imread(path)
    if resize_ratios != (1,1):
        image = cv2.resize(image, None,fx=resize_ratios[0], fy=resize_ratios[1], interpolation=cv2.INTER_CUBIC)
    return image

def make_screen_shot(channel):
    #if channel not in channels:
    #    raise Exception("Channel not found")
    channel += 1
    call_string = f"http://{ip_address}/cgi-bin/api.cgi?cmd=Snap&channel={channel}&rs=wuuPhkmUCeI9WG7C&user={username}&password={password}"
    reolink_response = requests.get(call_string, stream = True)
    if reolink_response.status_code != 200:
        raise Exception("Request Failed!")
    return reolink_response

def open_video_stream(channel):
    video = cv2.VideoCapture(f"rtsp://{username}:{password}@{ip_address}//h264Preview_0{channel}_sub")
    return video


def convert_to_matrix(http_response):
    return np.asarray(bytearray(http_response.content), dtype = "uint8")

def convert_to_cv_object(http_response):
    return cv2.imdecode(convert_to_matrix(http_response), cv2.IMREAD_COLOR)





def main():
    pass


if __name__== '__main__':
    main()
