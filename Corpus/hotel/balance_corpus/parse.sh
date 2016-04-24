#!/bin/sh
# coding:utf8

i=0
count=0
while [[ $i -lt 3000 ]]; do
	filename="$1.$i.txt"
	i=$((i+1))
	dos2unix $filename
	iconv -f gb18030 -t utf-8 $filename > n.$filename 
	if [[ $? -ne 0 ]]; then
		rm n.$filename
		count=$((count+1))
	fi
done
echo $count
