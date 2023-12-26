import dlib
import cv2
from scipy.spatial import distance as dist
from imutils.video import VideoStream, FileVideoStream
from imutils import face_utils
import imutils
import argparse
import os
import numpy as np


def eye_aspect_ratio(eye):
    # 计算眼部垂直方向上的2组关键点的欧氏距离
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # 计算眼部水平方向上的1组关键点的欧氏距离
    C = dist.euclidean(eye[0], eye[3])

    # 计算EAR
    ear = (A + B) / (2.0 * C)

    return ear


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", default='model/shape_predictor_68_face_landmarks.dat',
                help="path to facial landmark predictor")
ap.add_argument("-l", "--list", type=str, default="E:/Mead/Test",
                help="path of videos")
args = vars(ap.parse_args())
blink_interval = []
int_dir = "vec7/interval.npy"

EYE_AR_THRESH = 0.21
EYE_CON_THRESH = 3
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

for root, dirs, files in os.walk(args["list"]):
        for file in files:
            video_path = os.path.join(root, file)
            print(video_path)
            vs = FileVideoStream(video_path).start()
            fileStream = True
            frame_counter = 0
            blink_counter = 0
            prev_frame = 0
            current_frame = 0
            interval = []
            while vs.more():
                current_frame += 1
                frame = vs.read()
                if frame is None:
                    break
                frame = imutils.resize(frame, width=1920, height=1080)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rects = detector(gray, 0)

                for rect in rects:
                    shape = predictor(gray, rect)
                    shape = face_utils.shape_to_np(shape)

                    leftEye = shape[lStart:lEnd]
                    rightEye = shape[rStart:rEnd]
                    leftEAR = eye_aspect_ratio(leftEye)
                    rightEAR = eye_aspect_ratio(rightEye)
                    ear = (leftEAR + rightEAR) / 2.0
                    if ear < EYE_AR_THRESH:
                        frame_counter += 1
                    else:
                        if frame_counter >= EYE_CON_THRESH:
                            print("blink_gaze detected!", current_frame - prev_frame)
                            blink_counter += 1
                            interval.append(current_frame - prev_frame)
                            prev_frame = current_frame
                        frame_counter = 0

            vs.stop()
            print(interval)
            blink_interval.extend(interval)

blink_interval = np.array(blink_interval)
np.save(int_dir, blink_interval)

