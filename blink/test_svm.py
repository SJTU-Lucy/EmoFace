from sklearn import svm
from sklearn.externals import joblib
import numpy as np 


def calculate_accuracy(result, labels):
	correct_counter = 0
	for i in range(len(result)):
		if res[i] == labels[i]:
			correct_counter += 1
	total_num = len(result)
	return float(correct_counter) / float(total_num)


test_open_txt = open('test_open.txt', 'rb')
test_close_txt = open('test_close.txt', 'rb')

test = []
labels = []

print('Reading test_open.txt...')
line_ctr = 0
temp = []
for txt_str in test_open_txt.readlines():
	data = float(txt_str)
	if line_ctr <= 12:
		line_ctr += 1
		temp.append(data)
	elif line_ctr == 13:
		# print(temp)
		# print(len(temp))
		test.append(temp)
		labels.append(0)
		temp = []
		line_ctr = 1
		temp.append(data)


print('Reading test_close.txt...')
line_ctr = 0
temp = []
for txt_str in test_close_txt.readlines():
	data = float(txt_str)

	if line_ctr <= 12:
		line_ctr += 1
		temp.append(data)
	elif line_ctr == 13:
		test.append(temp)
		labels.append(1)
		temp = []
		line_ctr = 1
		temp.append(data)

for i in range(len(labels)):
	print("{0} --> {1}".format(test[i], labels[i]))

clf = joblib.load("ear_svm.m")

print('testing the svm model')
res = clf.predict(test)
print(res)
print(labels)

print('Accuracy:{0}'.format(calculate_accuracy(res, labels)))
