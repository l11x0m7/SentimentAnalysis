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
NEWS='../Corpus/news'

AMAZON_ELEC_CORPUS=${AMAZON}/elec.txt
AMAZON_BOOK_CORPUS=${AMAZON}/book.txt
DOUBAN_MEIRENYU_CORPUS=${DOUBAN}/meirenyu_final.txt
HOTEL_CORPUS=${HOTEL}/file/hotel_final.txt
NEWS_CORPUS=${NEWS}/news_final.txt

AMAZON_ELEC_SCORE=${AMAZON}/elec_score.txt
AMAZON_BOOK_SCORE=${AMAZON}/book_score.txt
DOUBAN_MEIRENYU_SCORE=${DOUBAN}/meirenyu_score.txt
HOTEL_SCORE=${HOTEL}/file/hotel_score.txt
NEWS_SCORE=${NEWS}/news_score.txt

while read -p 'Word Segment and POS Tagging?(y/n)' ans ;do
	case $ans in
		y*|Y* )
		read -p 'Please input the filepath: ' filepath
		read -p 'Please input the savepath: ' savepath
		read -p 'WS type(LTP or jieba): ' wstype
		read -p 'Which line would you like to start from?(at least 1) ' startline
		read -p 'How would you like to write to file?(w or a) ' addtype
		if [[ "x$filepath" == "x" || "x$savepath" == "x" || "x$wstype" == "x" || "x$startline" == "x" || "x$addtype" == "x" ]]; then
			echo 'Error!'
			exit 1
		fi
		python WSAndPOST.py \
		--filepath=$filepath \
		--savepath=$savepath \
		--wstype=$wstype \
		--startline=$startline \
		--addtype=$addtype
		echo "WS and POST done!"
		break
		;;
	esac
	case $ans in
		''|n*|N* )
		break
		;;
	esac
done

read -p 'Which file you want to test?(e/b/d/h/n)' f
read -p 'Which method would you like to use?(ML or SD) ' ans
# for test
case $f in 
	e )
	filepath=$AMAZON_ELEC_CORPUS
	;;
esac
case $f in 
	b )
	filepath=$AMAZON_BOOK_CORPUS
	;;
esac
case $f in 
	d )
	filepath=$DOUBAN_MEIRENYU_CORPUS
	;;
esac
case $f in 
	h )
	filepath=$HOTEL_CORPUS
	;;
esac
case $f in 
	n )
	filepath=$NEWS_CORPUS
	;;
esac

case $ans in
	ml|ML )
	read -p 'Which type of ML would you like to use?(SVM/DT/MB/NB/NN/LR) ' mtype
	read -p 'Which kind of feature gram would you like to use?(one/two/three) ' fnum
	python MLMethod.py \
	--filepath=$filepath \
	--classifiertype=$mtype \
	--featuregram=$fnum
	;;
esac

case $ans in
	sd|SD )
	# read -p 'Would you like to use the bi-freq dictionary?(k/j/d/h/n) ' scorepath
	# for test
	case $f in e )
		scorepath=$AMAZON_ELEC_SCORE
		;;
	esac
	case $f in b )
		scorepath=$AMAZON_BOOK_SCORE
		;;
	esac
	case $f in d )
		scorepath=$DOUBAN_MEIRENYU_SCORE
		;;
	esac
	case $f in h )
		scorepath=$HOTEL_SCORE
		;;
	esac
	case $f in n )
		scorepath=$NEWS_SCORE
		;;
	esac
	python DictMethod.py \
	--filepath=$filepath \
	--scorepath=$scorepath
	;;
esac

