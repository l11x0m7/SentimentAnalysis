# -*- encoding=utf-8 -*-
import sys 
import os
import json
import time
import math
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
				item = line.strip().decode()
				inverse_words.add(item)

	# 停用词：融合网络停用词、哈工大停用词、川大停用词
	stop_words = set()
	if getSW:
		with open(u'../Dict/StopWords/file/中文停用词库.txt') as fr:
			for line in fr:
				item = line.strip().decode()
				stop_words.add(item)
		with open(u'../Dict/StopWords/file/哈工大停用词表.txt') as fr:
			for line in fr:
				item = line.strip().decode()
				stop_words.add(item)
		with open(u'../Dict/StopWords/file/四川大学机器智能实验室停用词库.txt') as fr:
			for line in fr:
				item = line.strip().decode()
				stop_words.add(item)
		with open(u'../Dict/StopWords/file/百度停用词列表.txt') as fr:
			for line in fr:
				item = line.strip().decode()
				stop_words.add(item)
		with open(u'../Dict/StopWords/file/stopwords_net.txt') as fr:
			for line in fr:
				item = line.strip().decode()
				stop_words.add(item)
		with open(u'../Dict/StopWords/file/stopwords_net2.txt') as fr:
			for line in fr:
				item = line.strip().decode()
				stop_words.add(item)
		with open(u'../Dict/StopWords/file/amazon_stopwords.txt') as fr:
			for line in fr:
				item = line.strip().decode()
				stop_words.add(item)

	# 程度词：HowNet的程度词，most：6；very：5；more：4；shao：3；insuf：2
	degree_words = dict()
	if getDegree:
		with open(u'../Dict/DegreeWords/most.txt') as fr:
			for line in fr:
				degree_words[line.strip().decode()] = 1.0
		with open(u'../Dict/DegreeWords/very.txt') as fr:
			for line in fr:
				degree_words[line.strip().decode()] = 0.8
		with open(u'../Dict/DegreeWords/more.txt') as fr:
			for line in fr:
				degree_words[line.strip().decode()] = 0.6
		with open(u'../Dict/DegreeWords/shao.txt') as fr:
			for line in fr:
				degree_words[line.strip().decode()] = 0.4
		with open(u'../Dict/DegreeWords/insuf.txt') as fr:
			for line in fr:
				degree_words[line.strip().decode()] = 0.2


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
			if word.strip() == '':
				continue
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
def SentenceSentiment(sentence, inverse_words, degree_words, sentiment_words, word_score=dict(), weight=0.2):
	sent_score = 0.0
	no_sent_score = 0.0
	inverse_pos = 0
	degree_pos = 0
	inverse_list = list()
	degree_list = list()
	last_senti_word = -1
	# 记录情感词出现在一句话中的次数
	sent_word_num = 0
	for i, word in enumerate(sentence):
		if word in sentiment_words:
			sent_word_num += 1
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
			sent_score = sent_score + pos_weight * inverse_weight * degree_weight * senti_polarity
		elif word in word_score:
			no_sent_score += word_score[word] * weight
	if sent_word_num == 0:
		return no_sent_score
	else:
		return sent_score

# 篇章情感判定
# params:
# paragraph：已分词与词性标注的篇章
def ParagraphSentiment(paragraph, inverse_words, degree_words, sentiment_words, word_score=dict(), weight=0.2):
	ws_para, wt_para = SplitWSAndWT(paragraph)
	score = 0.0
	for sentence in ws_para:
		score += SentenceSentiment(sentence, inverse_words, degree_words, sentiment_words, word_score, weight)
	return score/len(ws_para)

# 对文件进行情感判定，统计指标
# filepath：输入文件路径
# threshold：score的判决门限
# mode=0:不加入共现频率的词得分；mode=1:加入共现频率的词得分
# scorepath:共现频率词得分列表
def CorpusSentiment(filepath, inverse_words, degree_words, sentiment_words, threshold=0, scorepath=None, weight=0.2):
	word_score = dict()
	if scorepath:
		with open(scorepath) as fr:
			for line in fr:
				items = line.strip().split('\t')
				word_score[items[0].decode()] = float(items[1])

	real_hypothesize_list = list()
	with open(filepath, 'r') as fr:
		for line in fr:
			each = line.strip().split('\t')
			paragraph = each[0]
			real_polar = int(each[1])
			score = ParagraphSentiment(paragraph, inverse_words, degree_words, sentiment_words, word_score, weight)
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

	# print '查准率：%f\n查全率：%f\nF值：%f\n识别度：%f\n准确率：%f' % (precision, recall, F_measure, recognize_rate, accuracy)
	return (round(precision, 4), round(recall, 4), round(F_measure, 4), round(recognize_rate, 4), round(accuracy, 4), round(recognize_rate*accuracy, 4))


"""---下面是基于词的共现频率的情感分析---"""

# 基于共现频率的词的得分
def BiFreqGenerator(filepath, savepath, stop_words, inverse_words, degree_words, sentiment_words, calcway='doc'):
	no_use_word_list = stop_words | inverse_words | set(degree_words)
	word2review = dict()
	review2word = dict()
	real_sentiment_words = list()
	total_words = 0
	total_docs = 0
	with open(filepath) as fr:
		for i, line in enumerate(fr.readlines()):
			items = line.strip().split('\t')
			review = items[0]
			ws_para, wt_para = SplitWSAndWT(review)
			total_docs += 1
			for i, sentence in enumerate(ws_para):
				for j, word in enumerate(sentence):
					if word not in no_use_word_list and wt_para[i][j] in ['a', 'v']:
						total_words += 1
						word2review.setdefault(word, dict())
						word2review[word].setdefault(i, 0)
						word2review[word][i] += 1
						review2word.setdefault(i, list())
						review2word[i].append(word)
						if word in sentiment_words:
							real_sentiment_words.append(word)

	word_similarity = dict()
	for i, sent_word in enumerate(real_sentiment_words):
		if sent_word in word2review:
			review_list = word2review[sent_word]
			# 按词频计算情感词出现概率
			if calcway == 'word':
				P_sent = sum([wordfreq for id, wordfreq in review_list.iteritems()])/float(total_words)
			# 按出现文档数计算情感词出现概率
			elif calcway == 'doc':
				P_sent = len(review_list)/float(total_docs)
			cowords = set()
			for review_id in review_list:
				words = review2word[review_id]
				cowords |= set(words)
			for word in cowords:
				uni_freq = 0
				# 按词频计算共现词出现概率
				if calcway == 'word':
					P_word = sum([wordfreq for id, wordfreq in word2review[word].iteritems()])/float(total_words)
				# 按出现文档数计算共现词出现概率
				elif calcway == 'doc':
					P_word = len(word2review[word])/float(total_docs)
				if word != sent_word:
					uni_review_id = set(review_list) & set(word2review[word])
					sent_score = sentiment_words[sent_word]
					word_similarity.setdefault(word, 0.0)
					P_uni = 0.0
					if calcway == 'doc':
						uni_freq = len(uni_review_id)
						# 按出现文档数计算情感词和共现词同时出现的概率
						P_uni = uni_freq/float(total_docs)
						
					elif calcway == 'word':
						for review_id in uni_review_id:
							freq = word2review[word][review_id]
							# if word2review.has_key(sent_word) and word2review[sent_word].has_key(review_id):
							uni_freq += min(freq, word2review[sent_word][review_id])
							# else:
								# uni_freq += freq
						# 按词频计算情感词和共现词同时出现的概率
						P_uni = float(uni_freq)/total_words
					
					delta_score = sent_score * (P_uni)
					word_similarity[word] += delta_score
		# print 'Finished %d, total %d' % (i+1, len(real_sentiment_words))

	# for word, score_list in word_similarity.iteritems():
	# 	word_similarity[word] = sum(score_list)

	max_score = float('-inf')
	min_score = float('inf')

	for word, score in word_similarity.iteritems():
		max_score = max(abs(score), max_score)
		min_score = min(abs(score), min_score)

	for word, score in word_similarity.iteritems():
		nominal_score = (score - min_score) / (max_score - min_score)
		word_similarity[word] = nominal_score

	word_similarity = sorted(word_similarity.iteritems(), key=lambda kk:kk[1], reverse=True)
	with open(savepath, 'w') as fw:
		for word, score in word_similarity:
			fw.write(word.encode() + '\t' + str(score) + '\n')
	

if __name__ == '__main__':
	reload(sys)
	sys.setdefaultencoding('utf-8')

	parser = ArgumentParser(description='dictmethod')
	parser.add_argument('--filepath', dest='filepath')
	parser.add_argument('--scorepath', dest='scorepath')
	args = parser.parse_args()


	if not args.scorepath:
		# 基于规则的情感分析
		inverse_words, stop_words, degree_words, sentiment_words = LoadDictionary()
		CorpusSentiment(args.filepath, inverse_words, degree_words, sentiment_words, 0)

	else:
		# 加入共现频率的情感分析
		res = list()
		inverse_words, stop_words, degree_words, sentiment_words = LoadDictionary()
		if not os.path.exists(args.scorepath):
			BiFreqGenerator(args.filepath, args.scorepath, stop_words, inverse_words, degree_words, sentiment_words, 'doc')
		for i in range(5, -1, -1):
			res.append(CorpusSentiment(args.filepath, inverse_words, degree_words, sentiment_words, 0, args.scorepath, 0.2*i))

		res = zip(*res)
		for i in res:
			for j in i:
				print '%.4f' % j













