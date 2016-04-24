#!/bin/sh
# -*- encoding=utf-8 -*-
if [[ "$1x" == "x" ]]; then
	LoadDate=`date +%Y%m%d`
else
	LoadDate=$1
fi

# 获取当前字典文件夹路径
EMOJI='../Dict/Emoji'
INVERSE='../Dict/InverseWords'
NET='../Dict/NetWords'
SENTIMENT='../Dict/SentimentWords'
STOP='../Dict/StopWords'

# 获取当前语料路径
AMAZON='../Corpus/amazon'
DOUBAN='../Corpus/douban'
HOTEL='../Corpus/hotel'

while read -p 'Word Segment and POS Tagging?(y/n)' ans ;do
	case $ans in
		y*|Y* )
		read -p 'Please input the filepath: ' filepath
		read -p 'Please input the savepath: ' savepath
		read -p 'WS type(LTP or jieba): ' wstype
		read -p 'Which line would you like to start from?(at least 1) ' startline
		read -p 'How would you like to write to file?(w or a) ' addtype
		if [ "x$filepath" == "x" -o "x$savepath" == "x" -o "x$wstype" == "x" -o "x$startline" == "x" -o "x$addtype" == "x"]; then
			echo 'Error!'
			exit 1
		fi
		python WSAndPOST.py \
		--filepath=$filepath \
		--savepath=$savepath \
		--wstype=$wstype \
		--startline=$startline \
		--addtype=$addtype
		;;
	esac
	case $ans in
		''|n*|N* )
		break
		;;
	esac
done

read -p 'Which file you want to test?(filepath) ' filepath
read -p 'Which method would you like to use?(ML or SD) ' ans
case $ans in
	ml|ML )
	read -p 'Which type of ML would you like to use?(SVM/DT/NB/GBDT/LR) ' mtype
	read -p 'Which kind of feature gram would you like to use?(one/two/three) ' fnum
	python MLMethod.py \
	--filepath=$filepath \
	--classifiertype=$mtype \
	--featuregram=$fnum
	;;
esac
case $ans in
	sd|SD )
	python DictMethod.py \
	--filepath=$filepath
	;;
esac

