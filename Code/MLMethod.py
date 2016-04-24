# -*- encoding:utf-8 -*-
# 考虑GBDT和NN（DN）
import sys
import json
import nltk
from random import shuffle
import numpy as np
import scipy as sp
from argparse import ArgumentParser
from nltk import BigramCollocationFinder
from nltk import TrigramCollocationFinder
from nltk import BigramAssocMeasures
from nltk import TrigramAssocMeasures
from nltk import FreqDist, ConditionalFreqDist
from tabulate import tabulate

import sklearn
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
"""
	功能：利用机器学习方法进行情感分析
	数据预处理：分词、去停用词
	特征表示：特征向量空间VSM，选择1gram、1gram和2gram搭配作为特征、1g+2g+3g作为特征
	特征维度：6000（可调整）
	特征权重：布尔型（词集模型）
	特征选择：互信息PMI、卡方统计CHI、文档频率DF
	特征提取：LSI、PCA、FDA
	分类器：SVM（线性核）、NB（伯努利二元朴素贝叶斯）、DT（分类决策树，ID3,CART）、LR
	测试指标：P、R、F1、Accuracy
"""

# 载入停用词表
def LoadStopWords():
	# 停用词：融合网络停用词、哈工大停用词、川大停用词
	stop_words = set()
	with open(u'../Dict/StopWords/file/中文停用词库.txt') as fr:
		for line in fr:
			item = line.strip()
			stop_words.add(item)
	with open(u'../Dict/StopWords/file/哈工大停用词表.txt') as fr:
		for line in fr:
			item = line.strip()
			stop_words.add(item)
	with open(u'../Dict/StopWords/file/四川大学机器智能实验室停用词库.txt') as fr:
		for line in fr:
			item = line.strip()
			stop_words.add(item)
	with open(u'../Dict/StopWords/file/百度停用词列表.txt') as fr:
		for line in fr:
			item = line.strip()
			stop_words.add(item)
	with open(u'../Dict/StopWords/file/stopwords_net.txt') as fr:
		for line in fr:
			item = line.strip()
			stop_words.add(item)
	with open(u'../Dict/StopWords/file/stopwords_net2.txt') as fr:
		for line in fr:
			item = line.strip()
			stop_words.add(item)
	return stop_words

# 将输入的已分词和词性标注的篇章进行分离，分别给出分词结果和词性标注结果
def SplitWSAndWT(bothParagraph):
	bothParagraph = json.loads(bothParagraph)
	ws_para = list()
	wt_para = list()
	for sentence in bothParagraph:
		ws_sentence = list()
		wt_sentence = list()
		for each in sentence:
			word, tag = each
			ws_sentence.append(word)
			wt_sentence.append(tag)
		ws_para.append(ws_sentence)
		wt_para.append(wt_sentence)
	return ws_para, wt_para

# 对输入的已分词和标注的数据进行预处理
def PreProcess(bothParagraph):
	# 获取分词部分的文本
	ws_para, wt_para = SplitWSAndWT(bothParagraph)

	# 去停用词，获取文档的词列表
	stop_words = LoadStopWords()
	stop_words.add('')
	stop_words.add(' ')
	cleared_wordlist = list()
	for sentence in ws_para:
		for word in sentence:
			if word not in stop_words:
				cleared_wordlist.append(word)
	return cleared_wordlist

# 词语特征（1-gram），word_list是整个语料的词表
def UniGramFeature(word_list):
	return dict([(word, True) for word in word_list])
	# END

# 2-gram，同上
def BiGramFeature(word_list, method=BigramAssocMeasures.chi_sq, n=2000):
	# 防止由于wordlist中只包含一种词而造成调用nbest引起的错误
	if len(set(word_list)) != 1:
		bigram_list = BigramCollocationFinder.from_words(word_list)
		top_bigram = bigram_list.nbest(method, n)
		return UniGramFeature(top_bigram)
	else:
		return UniGramFeature([])

# 3-gram，同上
def TriGramFeature(word_list, method=TrigramAssocMeasures.chi_sq, n=1000):
	trigram_list = TrigramCollocationFinder.from_words(word_list)
	top_trigram = trigram_list.nbest(method, n)
	return UniGramFeature(top_trigram)

# 1-gram + 2-gram
def Mixup2Feature(word_list, bi_method=BigramAssocMeasures.chi_sq,\
 bi_n=2000):
	# 防止由于wordlist中只包含一种词而造成调用nbest引起的错误
	if len(set(word_list)) != 1:
		bigram_list = BigramCollocationFinder.from_words(word_list)
		# print json.dumps(word_list, ensure_ascii=False)
		top_bigram = bigram_list.nbest(bi_method, bi_n)
		return UniGramFeature(word_list + top_bigram)
	else:
		return UniGramFeature(word_list)

# 1-gram + 2-gram + 3-gram
def Mixup3Feature(word_list, bi_method=BigramAssocMeasures.chi_sq,\
 bi_n=2000, tri_method=TrigramAssocMeasures.chi_sq, tri_n=1000):
	# 防止由于wordlist中只包含一种词而造成调用nbest引起的错误
	if len(set(word_list)) != 1:
		bigram_list = BigramCollocationFinder.from_words(word_list)
		top_bigram = bigram_list.nbest(bi_method, bi_n)
		trigram_list = TrigramCollocationFinder.from_words(word_list)
		top_trigram = trigram_list.nbest(tri_method, tri_n)
		return UniGramFeature(word_list + top_bigram + top_trigram)
	else:
		trigram_list = TrigramCollocationFinder.from_words(word_list)
		top_trigram = trigram_list.nbest(tri_method, tri_n)
		return UniGramFeature(word_list + top_trigram)

# 特征选择，默认用卡方统计
# params:
# type
# one:仅用词作为特征
# two:用词和二元组作为特征
# three:用词+二元组+三元组作为特征
# n:特征维数
def FeatureChoose(pos_wordlist, neg_wordlist, method=BigramAssocMeasures.chi_sq, featuregram='one', n=6000):
	pos_feature = list()
	neg_feature = list()
	pos_all_words = list()
	neg_all_words = list()
	# pos_all_feature = dict()
	# neg_all_feature = dict()
	if featuregram == 'one':
		for each in pos_wordlist:
			cur = UniGramFeature(each)
			pos_feature.append(cur)
			# pos_all_feature.update(cur)
			pos_all_words.extend(cur)
		for each in neg_wordlist:
			cur = UniGramFeature(each)
			neg_feature.append(cur)
			# neg_all_feature.update(cur)
			neg_all_words.extend(cur)
	elif featuregram == 'two':
		for each in pos_wordlist:
			cur = Mixup2Feature(each)
			pos_feature.append(cur)
			# pos_all_feature.update(cur)
			pos_all_words.extend(cur)
		for each in neg_wordlist:
			cur = Mixup2Feature(each)
			neg_feature.append(cur)
			# neg_all_feature.update(cur)
			neg_all_words.extend(cur)
	elif featuregram == 'three':
		for each in pos_wordlist:
			cur = Mixup3Feature(each)
			pos_feature.append(cur)
			# pos_all_feature.update(cur)
			pos_all_words.extend(cur)
		for each in neg_wordlist:
			cur = Mixup3Feature(each)
			neg_feature.append(cur)
			# neg_all_feature.update(cur)
			neg_all_words.extend(cur)
	else:
		return []

	fd = FreqDist()
	cfd = ConditionalFreqDist()
	for word in pos_all_words:
		fd[word] += 1
		cfd['pos'][word] += 1
	for word in neg_all_words:
		fd[word] += 1
		cfd['neg'][word] += 1
	pos_N = cfd['pos'].N()
	neg_N = cfd['neg'].N()
	N = fd.N()
	score_list = dict()
	for word, freq in fd.iteritems():
		pos_score = BigramAssocMeasures.chi_sq(cfd['pos'][word], (freq, pos_N), N)
		neg_score = BigramAssocMeasures.chi_sq(cfd['neg'][word], (freq, neg_N), N)
		score_list[word] = pos_score + neg_score

	best_topwords = sorted(score_list.iteritems(), key=lambda kk:kk[1], reverse=True)
	# print json.dumps(best_topwords[-100:-1], ensure_ascii=False)
	best_topwords = best_topwords[:n]
	# print json.dumps(best_topwords[:100], ensure_ascii=False)
	best_topwords = set(word for word, freq in best_topwords)
	return pos_feature, neg_feature, best_topwords

# 将篇章的词列表转化为VSM，特征使用筛选的topn个词（词组）
def VSMTagging(word_list, best_topwords):
	return dict([(word, True) for word in word_list if word in best_topwords])
	# END


# 绘制PR曲线
def DrawPrecisionRecallCurve(truth, pred):
	from sklearn import metrics
	from matplotlib import pyplot as plt
	precision, recall, thresh = metrics.precision_recall_curve(truth, pred)
	ax = plt.subplot(111)
	ax.plot(precision, recall)
	plt.show()



# 分类器分类、性能测试
# @params:
# @mode：如果mode=test，用的是train+devtest，如果mode=final，用的是train+test
def AccuracyByClassifier(classifier_model, pos_wordlist, neg_wordlist, mode='normal', knum=5):
	# 进行分类
	if classifier_model in ('svm', 'SVM'):
		# 线性核的SVM
		classifier_model = LinearSVC()
	elif classifier_model in ('mb', 'MB'):
		# 多项式朴素贝叶斯
		classifier_model = MultinomialNB()
	elif classifier_model in ('bb', 'BB'):
		# 伯努利朴素贝叶斯
		classifier_model = BernoulliNB()
	elif classifier_model in ('dt', 'DT'):
		# 决策树
		classifier_model = DecisionTreeClassifier(criterion='entropy')
	elif classifier_model in ('gbdt', 'GBDT'):
		# 梯度提升决策树GBDT
		classifier_model = GradientBoostingClassifier()
	else:
		# 默认用LR
		classifier_model = LogisticRegression()

	classifier = SklearnClassifier(classifier_model)
	tp = fp = tn = fn = 0
	if mode=='k-cross':
		all_wordlist = pos_wordlist + neg_wordlist
		shuffle(all_wordlist)
		precision = recall = F_measure = accuracy = 0.0
		real = list()
		pred = list()
		for i in range(knum):
			piece_len = int(len(all_wordlist)/knum)
			train_set = all_wordlist[:piece_len*i] + all_wordlist[piece_len*(i+1):]
			test_set = all_wordlist[piece_len*i:piece_len*(i+1)]
			classifier.train(train_set)
			for each in test_set:
				pre = classifier.classify(each[0])
				real.append(int(each[1]))
				pred.append(int(pre))
				if int(each[1]) == int(pre) and int(each[1]) == 1:
					tp += 1
				elif int(each[1]) == int(pre) and int(each[1]) == 0:
					tn += 1
				elif int(each[1]) != int(pre) and int(each[1]) == 1:
					fn += 1
				elif int(each[1]) != int(pre) and int(each[1]) == 0:
					fp += 1
			precision += float(tp)/(tp+fp)
			recall += float(tp)/(tp+fn)
			F_measure += 2.0/((1/precision) + (1/recall))
			accuracy += float(tp + tn) / (tp + fp + tn + fn)

		# DrawPrecisionRecallCurve(real, pred)

		return (precision/knum, recall/knum, F_measure/knum, accuracy/knum)
	elif mode=='normal':
		# 分出训练集和测试集
		pos_len = len(pos_wordlist)
		neg_len = len(neg_wordlist)
		# shuffle数据集
		shuffle(pos_wordlist)
		shuffle(neg_wordlist)
		train_set = pos_wordlist[:int(0.7*pos_len)] + neg_wordlist[:int(0.7*neg_len)]
		# devtest_set = pos_wordlist[int(0.6*pos_len):int(0.7*pos_len)] + neg_wordlist[int(0.6*neg_len):int(0.7*neg_len)]
		test_set = pos_wordlist[int(0.7*pos_len):] + neg_wordlist[int(0.7*neg_len):]
		classifier.train(train_set)
		real = list()
		pred = list()
		for each in test_set:
			pre = classifier.classify(each[0])
			real.append(int(each[1]))
			pred.append(int(pre))
			if int(each[1]) == int(pre) and int(each[1]) == 1:
				tp += 1
			elif int(each[1]) == int(pre) and int(each[1]) == 0:
				tn += 1
			elif int(each[1]) != int(pre) and int(each[1]) == 1:
				fn += 1
			elif int(each[1]) != int(pre) and int(each[1]) == 0:
				fp += 1
		precision = float(tp)/(tp+fp)
		recall = float(tp)/(tp+fn)
		F_measure = 2.0/((1/precision) + (1/recall))
		accuracy = float(tp + tn) / (tp + fp + tn + fn)
		# DrawPrecisionRecallCurve(real, pred)
		return (precision, recall, F_measure, accuracy, classifier)
	else:
		return (0, 0, 0, 0, 0)

# 情感分析
def MLMethod(filepath, classifiertype='LR', featuregram='one', featuredim=6000, method=BigramAssocMeasures.chi_sq, testtype='normal', knum=5):
	with open(filepath) as fr:
		pos_wordlist = list()
		neg_wordlist = list()
		# 数据预处理，得到已去停用词的评论列表
		for line in fr:
			items = line.strip().split('\t')
			both_para = items[0]
			ws_words = PreProcess(both_para)
			if int(items[1]) == 1:
				pos_wordlist.append(ws_words)
				# pos_allwords.extend(ws_words)
			else:
				neg_wordlist.append(ws_words)
				# neg_allwords.extend(ws_words)

		# 特征选择
		pos_wordlist, neg_wordlist, best_topwords = \
		FeatureChoose(pos_wordlist, neg_wordlist, featuregram=featuregram, n=featuredim, method=method)

		# 对每个语料建立VSM，进行特征空间映射
		for i, each in enumerate(pos_wordlist):
			pos_wordlist[i] = (VSMTagging(each, best_topwords), 1)
		for i, each in enumerate(neg_wordlist):
			neg_wordlist[i] = (VSMTagging(each, best_topwords), 0)
		
		measure_list = AccuracyByClassifier(classifiertype, pos_wordlist, neg_wordlist, mode=testtype, knum=knum)
		print '查准率：%f\n查全率：%f\nF-测值：%f\n准确率：%f' % (measure_list[0], measure_list[1], measure_list[2], measure_list[3])
		return measure_list[3]

if __name__ == '__main__':
	reload(sys)
	sys.setdefaultencoding('utf-8')

	parser = ArgumentParser(description='machinelearningmethod')
	parser.add_argument('--filepath', dest='filepath')
	parser.add_argument('--classifiertype', dest='classifiertype')
	parser.add_argument('--featuregram', dest='featuregram')
	args = parser.parse_args()

	# For Project
	# MLMethod(args.filepath, args.classifiertype, args.featuregram)

	# For Test
	word_num = [500, 1000, 2000, 4000, 6000, 8000, 10000]
	feature_type = ['one', 'two', 'three']
	method_type = ['BB', 'MB', 'DT', 'SVM']
	table = list()
	headers = ['Method']
	headers.extend(word_num)
	for i in method_type:
		table.append([i])
		for j in word_num:
			table[-1].append(MLMethod('../Corpus/hotel/file/hotel_final.txt', i, 'one', j, BigramAssocMeasures.chi_sq, 'k-cross', 5)) 
	
	print tabulate(table, headers=headers)

	# MLMethod('../Corpus/hotel/file/hotel_final.txt', 'MB', 'one', 4000, BigramAssocMeasures.chi_sq, 'k-cross', 5)
	# MLMethod('../Corpus/hotel/file/hotel_final.txt', 'MB', 'one', 6000, BigramAssocMeasures.chi_sq, 'k-cross', 5)
	# MLMethod('../Corpus/hotel/file/hotel_final.txt', 'DT', 'two', 6000, BigramAssocMeasures.chi_sq, 'k-cross', 5)
	# MLMethod('../Corpus/hotel/file/hotel_final.txt', 'DT', 'three', 6000, BigramAssocMeasures.chi_sq, 'k-cross', 5)





