from __future__ import division
import pandas as pd 
import subprocess
import platform,os
import sklearn
import numpy as np

def evaluationBypandas_f1_acc(df, predicted_label):
	df['predicted_label'] = predicted_label
	tp = 0
	fp = 0
	tn = 0
	fn = 0
	m = 0
	for i in range(len(df['predicted_label'])):
		if df['predicted_label'][i] == df['flag'][i] and df['flag'][i] == 1:
			tp = tp + 1
		if df['predicted_label'][i] == df['flag'][i] and df['flag'][i] == 0:
			tn = tn + 1
		if df['predicted_label'][i] != df['flag'][i] and df['flag'][i] == 1:
			fp = fp + 1
		if df['predicted_label'][i] != df['flag'][i] and df['flag'][i] == 0:
			fn = fn + 1
	if tp == 0 and fn == 0:
		recall_rate = 0
	else:
		recall_rate = tp / (tp + fn)
	precision_rate = tp / (tp + fp)
	if tp == 0 and fp == 0:
		precision_rate = 0
	accuracy = (tp + tn) / (tp + fp + tn + fn)
	if precision_rate == 0 and recall_rate == 0:
		f1_score = 0
	else:
		f1_score = 2 * precision_rate * recall_rate / (precision_rate + recall_rate)
	return accuracy, f1_score