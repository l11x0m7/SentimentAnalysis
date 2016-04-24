# coding=utf-8

import os
import sys
# dirpath为要遍历的目录，filenum为目录下的文件数，loadfile为保存的文件路径
def corpusParser(dirpath, filenum, loadfile):
	if os.path.exists(loadfile):
		os.remove(loadfile)
	with open(loadfile, 'a') as fw:
		for i in range(filenum):
			filepath = dirpath + (r'/n.neg.%d.txt' % i)
			if not os.path.exists(filepath):
				continue
			with open(filepath, 'r') as fr:
				word = ""
				for each in fr:
					each = each.strip()
					if each != "":
						word = word + " " + each
				fw.write(word.strip().encode('utf-8')+'\n')

def TransCorpusToSets(filepath):
	corpus_sets = set()
	with open(filepath) as fr:
		for each in fr:
			item = each.strip()
			corpus_sets.add(item)
	corpus_sets = list(corpus_sets)
	for each in corpus_sets:
		print each.encode('utf-8')


if __name__ == '__main__':
	reload(sys)
	sys.setdefaultencoding('utf-8')
	dirpath = r'../Corpus/hotel/balance_corpus/ChnSentiCorp_htl_ba_6000/neg'
	loadfile = r'../Corpus/hotel/balance_corpus/ChnSentiCorp_htl_ba_6000/neg/neg.txt'
	# corpusParser(dirpath, 3000, loadfile)

	filepath = r'../Corpus/douban/meirenyu.txt'
	TransCorpusToSets(filepath)



