import numpy as np 
from sklearn import svm
import joblib

train_open_txt = open('vec7/train_open.txt', 'r')
train_close_txt = open('vec7/train_close.txt', 'r')

train = []
labels = []

print('Reading train_open.txt...')
line_ctr = 0
for txt_str in train_open_txt.readlines():
	temp = []
	# print(txt_str)
	datas = txt_str.strip()
	print(datas)
	datas = datas.replace('[', '')
	datas = datas.replace(']', '')
	datas = datas.split(',')
	print(datas)
	for data in datas:
		# print(data)
		data = float(data)
		temp.append(data)
	# print(temp)
	train.append(temp)
	labels.append(0)


print('Reading train_close.txt...')
line_ctr = 0
temp = []
for txt_str in train_close_txt.readlines():
	temp = []
	datas = txt_str.strip()
	datas = datas.replace('[', '')
	datas = datas.replace(']', '')
	datas = datas.split(',')
	for data in datas:
		data = float(data)
		temp.append(data)
	train.append(temp)
	labels.append(1)


for i in range(len(labels)):
	print("{0} --> {1}".format(train[i], labels[i]))

train_close_txt.close()
train_open_txt.close()

print(train)
print(labels)
clf = svm.SVC(C=0.8, kernel='linear', gamma=20, decision_function_shape='ovo')
clf.fit(train, labels)
joblib.dump(clf, "vec7/ear_svm.m")

print('predicting [[0.34, 0.34, 0.31, 0.32, 0.32, 0.32, 0.33]]')
res = clf.predict([[0.34, 0.34, 0.31, 0.32, 0.32, 0.32, 0.33]])
print(res)

print('predicting [[0.34, 0.34, 0.12, 0.14, 0.15, 0.32, 0.33]]')
res = clf.predict([[0.34, 0.34, 0.12, 0.14, 0.15, 0.32, 0.33]])
print(res)

print('predicting [[0.19, 0.18, 0.18, 0.19, 0.18, 0.18, 0.17]]')
res = clf.predict([[0.19, 0.18, 0.18, 0.19, 0.18, 0.18, 0.17]])
print(res)

