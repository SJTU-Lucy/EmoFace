#coding=utf-8  
import cv2
import dlib
import numpy as np
from scipy.spatial import distance
import os
from imutils import face_utils
from imutils.video import FileVideoStream
import joblib
import imutils
import argparse


VECTOR_SIZE = 7
def queue_in(queue, data):
	ret = None
	if len(queue) >= VECTOR_SIZE:
		ret = queue.pop(0)
	queue.append(data)
	return ret, queue

def eye_aspect_ratio(eye):
	# print(eye)
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", default='model/shape_predictor_68_face_landmarks.dat',
                help="path to facial landmark predictor")
ap.add_argument("-l", "--list", type=str, default="E:/MEAD/M003",
                help="path of videos")
args = vars(ap.parse_args())
shape_detector_path = 'model/shape_predictor_68_face_landmarks.dat'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_detector_path)

clf = joblib.load("vec7/ear_svm.m")

EYE_AR_CONSEC_FRAMES = 3

RIGHT_EYE_START = 37 - 1
RIGHT_EYE_END = 42 - 1
LEFT_EYE_START = 43 - 1
LEFT_EYE_END = 48 - 1

interval = []
int_dir = "vec7/interval_svm.npy"


for root, dirs, files in os.walk(args["list"]):
	for file in files:
		if not file.endswith(".mp4"):
			continue
		video_path = os.path.join(root, file)
		print(video_path)
		vs = FileVideoStream(video_path).start()
		fileStream = True
		frame_counter = 0
		blink_counter = 0
		prev_frame = 0
		current_frame = 0
		ear_vector = []
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
				points = face_utils.shape_to_np(shape) # convert the facial landmark (x, y)-coordinates to a NumPy array
				# points = shape.parts()
				leftEye = points[LEFT_EYE_START:LEFT_EYE_END + 1]
				rightEye = points[RIGHT_EYE_START:RIGHT_EYE_END + 1]
				leftEAR = eye_aspect_ratio(leftEye)
				rightEAR = eye_aspect_ratio(rightEye)
				ear = (leftEAR + rightEAR) / 2.0
				ret, ear_vector = queue_in(ear_vector, ear)
				if (len(ear_vector) == VECTOR_SIZE):
					input_vector = []
					input_vector.append(ear_vector)
					res = clf.predict(input_vector)

					if res == 1:
						frame_counter += 1
					else:
						if frame_counter >= EYE_AR_CONSEC_FRAMES:
							print("blink_gaze detected!", current_frame - prev_frame)
							blink_counter += 1
							interval.append(current_frame - prev_frame)
							prev_frame = current_frame
						frame_counter = 0

blink_interval = np.array(interval)
np.save(int_dir, blink_interval)

	

