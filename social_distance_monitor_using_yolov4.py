import matplotlib.pyplot as plt
import numpy as np
import darknet
import random
import math
import time
import cv2
import os

from itertools import combinations
from PIL import Image
from ctypes import *


def euclidean_distance(p1, p2):
    distance = math.sqrt(p1 ** 2 + p2 ** 2)
    return distance


def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cvDrawBoxes(detections, img):
    if len(detections) > 0:
        centroid_dict = dict()
        object_id = 0
        for label, confidence, bbox in detections:
            if label == 'person' and float(confidence) > 50:
                x, y, w, h = (bbox[0], bbox[1], bbox[2], bbox[3])
                print(f"Person # {object_id}, located: ({int(x)}, {int(y)}) " +
                      f"confidence: {confidence}")
                xmin, ymin, xmax, ymax = convertBack(float(x), float(y),
                                                     float(w), float(h))
                centroid_dict[object_id] = (int(x), int(y), xmin,
                                            ymin, xmax, ymax)
                object_id += 1

        red_zone_list = list()
        red_line_list = list()

        for (id1, p1), (id2, p2) in combinations(centroid_dict.items(), 2):
            dx, dy = p1[0] - p2[0], p1[1] - p2[1]
            distance = euclidean_distance(dx, dy)
            if distance < 60.0:
                if id1 not in red_zone_list:
                    red_zone_list.append(id1)
                    red_line_list.append(p1[0:2])
                if id2 not in red_zone_list:
                    red_zone_list.append(id2)
                    red_line_list.append(p2[0:2])
        for idx, box in centroid_dict.items():
            if idx in red_zone_list:
                cv2.rectangle(img, (box[2], box[3]), (box[4], box[5]),
                              (255, 0, 0), 2)
            else:
                cv2.rectangle(img, (box[2], box[3]), (box[4], box[5]),
                              (0, 255, 0), 2)
        amount_people = len(centroid_dict.keys())
        amount_bad_people = len(red_zone_list)

        print(f"\nTotal number of people {amount_people}\n" +
              f"Total number of people who break social distancing measure " +
              f"{amount_bad_people}\nPeople ids who break social " +
              f"distancing measure {red_zone_list}\n")

        risk_percentage = round((amount_bad_people / amount_people) * 100, 2)
        text = f"Risk Percentage: {str(risk_percentage)}%"
        location = (15, 30)
        cv2.putText(img, text, location, cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 0), 2, cv2.LINE_AA)

        for check in range(0, len(red_line_list) - 1):
            start_point = red_line_list[check]
            end_point = red_line_list[check + 1]
            check_line_x = abs(end_point[0] - start_point[0])
            check_line_y = abs(end_point[1] - start_point[1])
            if (check_line_x < 75) and (check_line_y < 25):
                cv2.line(img, start_point, end_point, (255, 0, 0), 2)
    return img, risk_percentage


netMain = None
metaMain = None
altNames = None


def YOLO(file_input):
    global metaMain, netMain, altNames
    configPath = "./cfg/yolov4.cfg"
    weightPath = "./yolov4.weights"
    metaPath = "./cfg/coco.data"
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath) + "`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath) + "`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath) + "`")

    network, class_names, class_colors = darknet.load_network(configPath,
                                                              metaPath,
                                                              weightPath,
                                                              batch_size=1)

    cap = cv2.VideoCapture(file_input)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    new_height, new_width = frame_height // 2, frame_width // 2
    file_output = file_input.replace('.mp4', '_out.avi')
    file_output = file_output.replace('Input', 'Output')
    out = cv2.VideoWriter(file_output, cv2.VideoWriter_fourcc(*"MJPG"), 10.0,
                          (new_width, new_height))

    darknet_image = darknet.make_image(new_width, new_height, 3)
    c = 0
    total_risk = 0
    while True:
        prev_time = time.time()
        ret, frame_read = cap.read()
        if (not ret) or (c > 150):  # Number of frames for the resulting video
            break  # If you want all the output comment it, but
        c += 1  # it will last a lot.
        print(f"** Frame # {c} **\n")
        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (new_width, new_height),
                                   interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())
        detections = darknet.detect_image(network, class_names, darknet_image,
                                          thresh=0.25)

        image, risk = cvDrawBoxes(detections, frame_resized)
        total_risk += risk
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        out.write(image)
        # print(1 / (time.time() - prev_time))

        img = Image.fromarray(image, 'RGB')
        plt.imshow(img)
        plt.show()
        img.close()

    total_risk = round(total_risk / c, 2)
    cap.release()
    out.release()
    print("** End of Detection **\n** The percentage risk for this video is " +
          f"about {total_risk}%\nPlease look at the output of the system" +
          " in the file located in Simulation_project/data/Output **")


file_input = '/content/Simulation_project/data/Input/OxfordTownCenter.mp4'
YOLO(file_input)
