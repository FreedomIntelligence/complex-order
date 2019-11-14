from __future__ import division
import pandas as pd 
import subprocess
import platform,os
import sklearn
import numpy as np
from sklearn.metrics import f1_score
qa_path="data/nlpcc-iccpol-2016.dbqa.testing-data"

def mrr_metric(group):
	group = sklearn.utils.shuffle(group,random_state =132)
	candidates=group.sort_values(by='score',ascending=False).reset_index()
	rr=candidates[candidates["flag"]==1].index.min()+1
	if rr!=rr:
		return 0
	return 1.0/rr
def map_metric(group):
	group = sklearn.utils.shuffle(group,random_state =132)
	ap=0
	candidates=group.sort_values(by='score',ascending=False).reset_index()
	correct_candidates=candidates[candidates["flag"]==1]
	if len(correct_candidates)==0:
		return 0
	for i,index in enumerate(correct_candidates.index):
		ap+=1.0* (i+1) /(index+1)
	#print( ap/len(correct_candidates))
	return ap/len(correct_candidates)

def evaluation_plus(modelfile, groundtruth=qa_path):
	answers=pd.read_csv(groundtruth,header=None,sep="\t",names=["question","answer","flag"],quoting =3)
	answers["score"]=pd.read_csv(modelfile,header=None,sep="\t",names=["score"],quoting =3)
	print( answers.groupby("question").apply(mrr_metric).mean())
	print( answers.groupby("question").apply(map_metric).mean())
def my_f1_score(y_true,y_pred):

	return f1_score(y_true,y_pred,average = 'macro')
def precision_recall_f1(prediction, ground_truth):
    num_same = sum(np.equal(prediction,ground_truth).astype('int'))
    print(num_same)

    if num_same == 0:
        return 0, 0, 0
    p = 1.0 * num_same / len(prediction)
    r = 1.0 * num_same / len(ground_truth)
    f1 = (2 * p * r) / (p + r)
    return p, r, f1
def evaluationBypandas_f1_acc(df, predicted, predicted_label):
	df["score"] = predicted
	df['predicted_label'] = predicted_label
	tp = 0
	fp = 0
	tn = 0
	fn = 0
	m = 0
	for i in range(len(df['score'])):
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
	accuracy = (tp + tn) / (tp + fp + tn + fn)
	if precision_rate == 0 and recall_rate == 0:
		f1_score = 0
	else:
		f1_score = 2 * precision_rate * recall_rate / (precision_rate + recall_rate)
	return accuracy

def eval(predicted,groundtruth=qa_path, file_flag=False):
	if  'Windows' in platform.system() and file_flag ==False:
		modelfile=write2file(predicted)
		evaluationbyFile(modelfile)
		return 

	if type(groundtruth)!= str :
		answers=groundtruth
	else:
		answers=pd.read_csv(groundtruth,header=None,sep="\t",names=["question","answer","flag"],quoting =3)
	answers["score"]=predicted
	mrr= answers.groupby("question").apply(mrr_metric).mean()
	map= answers.groupby("question").apply(map_metric).mean()
	return map,mrr
def evaluate(predicted,groundtruth):
	filename=write2file(predicted)
	evaluationbyFile(filename,groundtruth=groundtruth)
def write2file(datas,filename="train.QApair.TJU_IR_QA.score"):
	with open(filename,"w") as f:
		for data in datas:
			f.write(("%.10f" %data )+"\n")
	return filename


def evaluationbyFile(modelfile,resultfile="result.text",groundtruth=qa_path):
	cmd="test.exe " + " ".join([groundtruth,modelfile,resultfile])
	print( modelfile[19:-6]+":") # )
	subprocess.call(cmd, shell=True)
def evaluationBypandas(df,predicted):
	df["score"]=predicted
	mrr= df.groupby("question").apply(mrr_metric).mean()
	map= df.groupby("question").apply(map_metric).mean()
	return map,mrr
def precision_per(group):
	group = sklearn.utils.shuffle(group,random_state =132)
	candidates=group.sort_values(by='score',ascending=False).reset_index()
	rr=candidates[candidates["flag"]==1].index.min()
	if rr==0:
		return 1
	return 0
def precision(df,predicted):
	df["score"]=predicted
	precision = df.groupby("question").apply(precision_per).mean()
	return precision

def briany_test_file(df_test,  predicted=None,mode = 'test'):
	N = len(df_test)

	nnet_outdir = 'tmp/' + mode
	if not os.path.exists(nnet_outdir):
		os.makedirs(nnet_outdir)
	question2id=dict()
	for index,quesion in enumerate( df_test["question"].unique()):
		question2id[quesion]=index

	df_submission = pd.DataFrame(index=np.arange(N), columns=['qid', 'iter', 'docno', 'rank', 'sim', 'run_id'])
	df_submission['qid'] =df_test.apply(lambda row: question2id[row['question']],axis=1)
	df_submission['iter'] = 0
	df_submission['docno'] = np.arange(N)
	df_submission['rank'] = 0
	if  predicted is None:
		df_submission['sim'] = df_test['score']
	else:
		df_submission['sim'] = predicted
	df_submission['run_id'] = 'nnet'
	df_submission.to_csv(os.path.join(nnet_outdir, 'submission.txt'), header=False, index=False, sep=' ')

	df_gold = pd.DataFrame(index=np.arange(N), columns=['qid', 'iter', 'docno', 'rel'])
	df_gold['qid'] = df_test.apply(lambda row: question2id[row['question']],axis=1)
	df_gold['iter'] = 0
	df_gold['docno'] = np.arange(N)
	df_gold['rel'] = df_test['flag']
	df_gold.to_csv(os.path.join(nnet_outdir, 'gold.txt'), header=False, index=False, sep=' ')

if __name__ =="__main__":
	data_dir="data/"+"wiki"
	train_file=os.path.join(data_dir,"train.txt")
	test_file=os.path.join(data_dir,"test.txt")

	train=pd.read_csv(train_file,header=None,sep="\t",names=["question","answer","flag"],quoting =3)
	train["score"]=np.random.randn(len(train))
	briany_test_file(train)