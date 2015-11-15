import numpy as np


from util import *

from run_knn import *
import matplotlib.pyplot as plt

def correctPredictRate(target, predict):
	dimension = len(target)
	if dimension != len(predict):
		print ('wrong dimension')
		return
	correctNum = 0
	for i in range(dimension):
		if (target[i] == predict[i]):
			correctNum += 1 
	return (float(correctNum)/dimension)

def run():
	inputs_train, inputs_valid, inputs_test, target_train, target_valid, target_test = LoadData('digits.npz')

	predict_label_dict = {}
	for i in range(10):
		k = 1 + 2*i
		# predict_label_dict[k] = run_knn(k, inputs_train, target_train, inputs_valid)
		predict_label_dict[k] = run_knn(k, inputs_train, target_train, inputs_train)
		

	# plot config
	area = np.pi*(3)**2

	for k, predict_label in predict_label_dict.iteritems():
		# rate = correctPredictRate(targets_valid, predict_label)
		rate = correctPredictRate(target_train, predict_label)
		plt.scatter(k, rate, s=area, alpha=0.8)
	plt.show()

if __name__ == '__main__':
	run()