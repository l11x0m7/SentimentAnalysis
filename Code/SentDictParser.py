# -*- coding:utf-8 -*-
import sys
import xlrd

"""
	Function：
	通过文件解析，融合了SentimentWords目录下
	的所有情感词典到该目录下的two_emotions.txt文件
	Output Params：
	输出包含情感词和极性（1:positive，－1:negative）
"""

# 解析大连理工大学的情感词汇数据
def DUTParser():
	dut = xlrd.open_workbook(u'../Dict/SentimentWords/DUT/file/情感词汇本体.xlsx')
	sheet1 = dut.sheet_by_index(0)
	col1 = sheet1.col_values(0)
	col2 = sheet1.col_values(6)
	with open('../Dict/SentimentWords/DUT/two_motions.txt', 'w') as fw:
		for i, word in enumerate(col1):
			if i == 0:
				continue
			if int(float(col2[i])) == 1:
				dut_list.append((col1[i].decode(), 1))
			elif int(float(col2[i])) == 2:
				dut_list.append((col1[i].decode(), -1))
			else:
				continue
			content = '\t'.join(map(str, dut_list[-1]))
			fw.write(content + '\n')

# 知网情感词解析
def HowNetParser():
	with open(u'../Dict/SentimentWords/HowNet/file/正面情感词语（中文）.txt') as fr:
		for line in fr:
			hownet_list.append((line.strip().decode(), 1))

	with open(u'../Dict/SentimentWords/HowNet/file/正面评价词语（中文）.txt') as fr:
		for line in fr:
			hownet_list.append((line.strip().decode(), 1))

	with open(u'../Dict/SentimentWords/HowNet/file/负面情感词语（中文）.txt') as fr:
		for line in fr:
			hownet_list.append((line.strip().decode(), -1))

	with open(u'../Dict/SentimentWords/HowNet/file/负面评价词语（中文）.txt') as fr:
		for line in fr:
			hownet_list.append((line.strip().decode(), -1))

# 台湾大学情感词典解析
def NTUSDParser():
	with open(u'../Dict/SentimentWords/NTUSD/file/ntusd-positive.txt') as fr:
		for line in fr:
			ntusd_list.append((line.strip().decode(), 1))

	with open(u'../Dict/SentimentWords/NTUSD/file/ntusd-negative.txt') as fr:
		for line in fr:
			ntusd_list.append((line.strip().decode(), -1))

# 哈工大同义词词林解析，用来扩展情感词典
def HITParser():
	with open(u'../Dict/SentimentWords/HIT/file/哈工大信息检索研究中心同义词词林扩展版.txt') as fr:
		for line in fr:
			items = line.decode().strip().split(' ')
			hit_list.append(set(items[1:]))

def MergeDict():
	final_list = list()
	with open(u'../Dict/SentimentWords/two_motions.txt', 'w') as fw:
		for each in dut_list:
			if each not in final_list:
				if (each[0],-1*each[1]) not in final_list:
					final_list.append(each)
		for each in hownet_list:
			if each not in final_list:
				if (each[0],-1*each[1]) not in final_list:
					final_list.append(each)
		for each in ntusd_list:
			if each not in final_list:
				if (each[0],-1*each[1]) not in final_list:
					final_list.append(each)

		for each in final_list:
			content = '\t'.join(map(str,each))
			fw.write(content + '\n')


if __name__ == '__main__':
	reload(sys)
	sys.setdefaultencoding('utf-8')

	dut_list = list()
	DUTParser()

	hownet_list = list()
	HowNetParser()

	ntusd_list = list()
	NTUSDParser()

	hit_list = list()

	MergeDict()
