# -*- encoding=utf-8 -*-
import sys 
import os
import json
import time
from argparse import ArgumentParser
# 融合词性标注以及规则优化
"""
	功能：通过情感字典对情感进行分析
	过程：
	1）载入情感字典（双极性）、否定词、程度词
	2）分词（加载入网络热词）
	3）规则分析：寻找每句话的情感词、情感词前的程度词和否定词
	           如果程度词在否定词前，则为强负极性；若在之后，则是弱正极性
	           判断否定词个数，如果是奇数，则为负极性；若为偶数，则为正极性
	           对情感词前所有程度词进行统计，找出最大分值的程度词作为该情感词的修饰词
	输出：P、R、F1、识别度、Accuracy
"""

# 获取字典内容，包括否定词、程度词、情感词
def LoadDictionary(getInverse=True, getSW=True, getDegree=True, getSent=True):
	# 否定词
	inverse_words = set()
	if getInverse:
		with open('../Dict/InverseWords/inversewords.txt') as fr:
			for line in fr:
				item = line.strip()
				inverse_words.add(item)

	# 停用词：融合网络停用词、哈工大停用词、川大停用词
	stop_words = set()
	if getSW:
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

	# 程度词：HowNet的程度词，most：6；very：5；more：4；shao：3；insuf：2
	degree_words = dict()
	if getDegree:
		with open(u'../Dict/DegreeWords/most.txt') as fr:
			for line in fr:
				degree_words[line.strip().decode()] = 2.0
		with open(u'../Dict/DegreeWords/very.txt') as fr:
			for line in fr:
				degree_words[line.strip().decode()] = 1.5
		with open(u'../Dict/DegreeWords/more.txt') as fr:
			for line in fr:
				degree_words[line.strip().decode()] = 1.2
		with open(u'../Dict/DegreeWords/shao.txt') as fr:
			for line in fr:
				degree_words[line.strip().decode()] = 0.8
		with open(u'../Dict/DegreeWords/insuf.txt') as fr:
			for line in fr:
				degree_words[line.strip().decode()] = 0.4


	# 情感词典：包括positive和negative的，1表示正，－1表示负
	sentiment_words = dict()
	if getSent:
		with open(u'../Dict/SentimentWords/two_motions.txt') as fr:
			for line in fr:
				items = line.strip().split('\t')
				sentiment_words[items[0].decode()] = float(items[1])


	return inverse_words, stop_words, degree_words, sentiment_words

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

# 分析句子的情感，判定分数，给出结果
# params
# sentence:已分词的句子；
# inverse_words：否定集；
# degree_words：程度集
# sentiment_words：情感字典
# 规则：先找到情感词和该情感词对应的分值，然后从上一个情感词到这个情感词之间找出否定词、和程度词
#（1）否定词个数为奇数，则判定为否定；个数为偶数，则判定为肯定。
#（2）第一个程度词出现在第一个否定词之前，则判定为“强负极性”；出现在之后，则判定为“弱正极性”
#（3）对程度词，选取得分最高的词
def SentenceSentiment(sentence, inverse_words, degree_words, sentiment_words):
	score = 0.0
	inverse_pos = 0
	degree_pos = 0
	inverse_list = list()
	degree_list = list()
	last_senti_word = -1
	for i, word in enumerate(sentence):
		if word in sentiment_words:
			senti_polarity = sentiment_words[word]
			for j in range(last_senti_word + 1, i):
				if sentence[j] in inverse_words:
					inverse_list.append((j, -1))
				elif sentence[j] in degree_words:
					degree_list.append((j, degree_words[sentence[j]]))
			last_senti_word = i
			degree_weight = 1.0
			inverse_weight = 1.0
			pos_weight = 1.0
			if len(inverse_list) != 0 and len(degree_list) != 0:
				degree_weight = max(s[1] for s in degree_list)
				inverse_weight = -1.0 if len(degree_list)%2 == 1 else 1.0
				pos_weight = 0.5 if inverse_list[0] < degree_list[0] else -1.0
			elif len(inverse_list) != 0:
				inverse_weight = -1.0 if len(degree_list)%2 == 1 else 1.0
			elif len(degree_list) != 0:
				degree_weight = max(s[1] for s in degree_list)
			score = score + pos_weight * inverse_weight * degree_weight * senti_polarity
	return score

# 篇章情感判定
# params:
# paragraph：已分词与词性标注的篇章
def ParagraphSentiment(paragraph, inverse_words, degree_words, sentiment_words):
	ws_para, wt_para = SplitWSAndWT(paragraph)
	score = 0.0
	for sentence in ws_para:
		score += SentenceSentiment(sentence, inverse_words, degree_words, sentiment_words)
	return score/len(ws_para)

# 对文件进行情感判定，统计指标
# filepath：输入文件路径
# threshold：score的判决门限
def FileSentiment(filepath, inverse_words, degree_words, sentiment_words, threshold):
	real_hypothesize_list = list()
	with open(filepath, 'r') as fr:
		for line in fr:
			each = line.strip().split('\t')
			paragraph = each[0]
			real_polar = int(each[1])
			score = ParagraphSentiment(paragraph, inverse_words, degree_words, sentiment_words)
			if score == 0:
				real_hypothesize_list.append((real_polar, -2))
			else:
				real_hypothesize_list.append((real_polar, 0) if score < threshold else (real_polar, 1))
	tp = fp = tn = fn = 0
	for real, hypothesize in real_hypothesize_list:
		if hypothesize != -2:
			if real == hypothesize and real == 1:
				tp += 1
			elif real == hypothesize and real == 0:
				tn += 1
			elif real != hypothesize and real == 1:
				fn += 1
			elif real != hypothesize and real == 0:
				fp += 1

	precision = float(tp)/(tp+fp)
	recall = float(tp)/(tp+fn)
	F_measure = 2.0/((1/precision) + (1/recall))
	recognize_rate = float(tp + fp + tn + fn)/len(real_hypothesize_list)
	accuracy = float(tp + tn) / (tp + fp + tn + fn)

	print '查准率：%f\n查全率：%f\nF值：%f\n识别度：%f\n准确率：%f' % (precision, recall, F_measure, recognize_rate, accuracy)


if __name__ == '__main__':
	reload(sys)
	sys.setdefaultencoding('utf-8')

	parser = ArgumentParser(description='dictmethod')
	parser.add_argument('--filepath', dest='filepath')
	args = parser.parse_args()

	inverse_words, stop_words, degree_words, sentiment_words = LoadDictionary()
	FileSentiment(args.filepath, inverse_words, degree_words, sentiment_words, 0)














