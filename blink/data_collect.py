import numpy as np
import os
import dlib
import cv2
from scipy.spatial import distance
from imutils import face_utils
from imutils.video import VideoStream, FileVideoStream
import pickle
import random
import argparse

RADIUS = 3
def queue_in(queue, data):
	if len(queue) >= 2 * RADIUS + 1:
		queue.pop(0)
	queue.append(data)
	return queue


def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", default='model/shape_predictor_68_face_landmarks.dat',
				help="path to facial landmark predictor")
ap.add_argument("-l", "--list", type=str, default="E:/MEAD",
				help="path of videos")
args = vars(ap.parse_args())

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

cv2.namedWindow("frame", cv2.WINDOW_NORMAL)

RIGHT_EYE_START = 37 - 1
RIGHT_EYE_END = 42 - 1
LEFT_EYE_START = 43 - 1
LEFT_EYE_END = 48 - 1

print('Start Collecting Data')
print('Press b for blink frames .')
print('Press o for open frames.')
txt_open = open('vec7/train_open.txt', 'a')
txt_close = open('vec7/train_close.txt', 'a')
open_counter = 0
close_counter = 0
ear_vector = []
quit = False

for root, dirs, files in os.walk(args["list"]):
	for file in files:
		if not file.endswith(".mp4") or random.random() > 0.05:
			continue
		video_path = os.path.join(root, file)
		print(video_path)
		vs = FileVideoStream(video_path).start()
		fileStream = True
		ear_seq = []
		flag_seq = []

		while vs.more():
			frame = vs.read()
			if frame is None:
				break
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			rects = detector(gray, 0)
			for rect in rects:
				shape = predictor(gray, rect)
				points = face_utils.shape_to_np(shape)  # convert the facial landmark (x, y)-coordinates to a NumPy array
				leftEye = points[LEFT_EYE_START:LEFT_EYE_END + 1]
				rightEye = points[RIGHT_EYE_START:RIGHT_EYE_END + 1]
				leftEAR = eye_aspect_ratio(leftEye)
				rightEAR = eye_aspect_ratio(rightEye)
				ear = (leftEAR + rightEAR) / 2.0
				cv2.putText(frame, "EAR:{:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				cv2.imshow("frame", frame)
				key = cv2.waitKey(0)
				if key & 0xFF == ord("b"):
					ear_seq = queue_in(ear_seq, ear)
					flag_seq = queue_in(flag_seq, 1)
				elif key & 0xFF == ord("o"):
					ear_seq = queue_in(ear_seq, ear)
					flag_seq = queue_in(flag_seq, 2)
				elif key & 0xFF == ord("q"):
					quit = True
				else:
					ear_seq = queue_in(ear_seq, ear)
					flag_seq = queue_in(flag_seq, 0)
				if len(ear_seq) == 2 * RADIUS + 1:
					if flag_seq[RADIUS] == 1:
						txt_close.write(str(ear_seq))
						txt_close.write('\n')
						close_counter += 1
						print("close counter:", close_counter, ear_seq)
					elif flag_seq[RADIUS] == 2:
						txt_open.write(str(ear_seq))
						txt_open.write('\n')
						open_counter += 1
						print("open counter:", open_counter, ear_seq)
			if quit:
				break
		vs.stop()
		if quit:
			break
	if quit:
		break

txt_open.close()
txt_close.close()
cv2.destroyAllWindows()
