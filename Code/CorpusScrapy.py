# encoding=utf-8

import sys, os
import re
import urllib
import urllib2
import cookielib
import requests
import time
reload(sys)
sys.setdefaultencoding('utf-8')


# positive为好评数量，critical为差评数量，item_num为商品编号
def getReviewsFromAmazon(positive, critical, item_num):
	# item_num = re.findall(r'.*amazon.cn.*?/product/(.*?)\?.*?', url, re.S)[0]
	user_agent = 'Mozilla/4.0 (compatible; MSIE 5.5; Windows NT)'
	headers = { 'User-Agent' : user_agent }
	print 'The item number is: ', item_num
	for emotion in ['positive', 'critical']:
		if emotion == 'positive':
			pagenum = (positive-1)/10 + 1
		else:
			pagenum = (critical-1)/10 + 1
		review_tag = 1
		if os.path.exists('../Corpus/amazon/amazon_%s_%s.txt'% (item_num, emotion)):
			os.remove('../Corpus/amazon/amazon_%s_%s.txt'% (item_num, emotion))
			print '%s has been removed' % emotion
		with open('../Corpus/amazon/amazon_%s_%s.txt'% (item_num, emotion), 'a') as fw:
			i = 1
			while i <pagenum+1:
				url = 'http://www.amazon.cn/product-reviews/%s/ref=cm_cr_getr_d_paging_btm_1?ie=UTF8&showViewpoints=1&sortBy=recent&pageNumber=%d&filterByStar=%s' % (item_num, i, emotion)
				req = urllib2.Request(url, headers=headers)
				try:
					response = urllib2.urlopen(req)
					content = response.read()
					# unicode_content = content.decode('utf-8')
					pattern = r'<span class="a-size-base review-text">(.*?)</span></div>'
					reviews = re.findall(pattern, content, re.S)
					reviews = [review.replace('<br />', '\n').replace('\n', ' ') for review in reviews]
					for review in reviews:
						res = (str(review_tag) + '\t' + review.encode('utf8') + '\n')
						review_tag += 1
						fw.write(res)
					print '[%s finished] %.2f%%' % (emotion, float(i*100)/pagenum)
					i += 1
					time.sleep(0.1)

				except urllib2.URLError, e:
					print '[Error] can\'t get the page %d' % i
					print 'Retrying......'

def getReviewFromDouban(url, reviewnum, moviename):
	loginurl = 'https://www.douban.com/accounts/login'
	cookie = cookielib.CookieJar()
	opener = urllib2.build_opener(urllib2.HTTPCookieProcessor(cookie))
	params = {
		"form_email":"l11x0m7@gmail.com",
		"form_password":"lin62870502",
		"source":"index_nav" #没有的话登录不成功
	}
	#从首页提交登录
	response=opener.open(loginurl, urllib.urlencode(params))
	#验证成功跳转至登录页
	if response.geturl() == "https://www.douban.com/accounts/login":
  		html=response.read()
	  	#验证码图片地址
	  	imgurl=re.search('<img id="captcha_image" src="(.+?)" alt="captcha" class="captcha_image"/>', html)
	  	if imgurl:
	  		pic_url=imgurl.group(1)
	    	#将图片保存至同目录下
	    	res=urllib.urlretrieve(pic_url, 'v.jpg')
	    	#获取captcha-id参数
	    	captcha=re.search('<input type="hidden" name="captcha-id" value="(.+?)"/>' ,html)
	    	if captcha:
	      		vcode=raw_input('请输入图片上的验证码：')
	      		params["captcha-solution"] = vcode
	      		params["captcha-id"] = captcha.group(1)
	      		params["user_login"] = "登录"
	      	#提交验证码验证
	      	response=opener.open(loginurl, urllib.urlencode(params))
	      	''' 登录成功跳转至首页 '''
	      	if response.geturl() == "http://www.douban.com/":
	        	print 'Login success!'
				# user_agent = 'Mozilla/4.0 (compatible; MSIE 5.5; Windows NT)'
				# headers = { 'User-Agent' : user_agent }
                page_num = (reviewnum-1)/20 + 1
                with open(r'../Corpus/douban/%s.txt' % moviename, 'w') as fw:
					i = 0
					while i < page_num:
						real_url = url + str(20*i)
						# req = urllib2.Request(real_url, headers=headers)
						try:
							response = opener.open(real_url)
							content = response.read()
							# unicode_content = content.decode('utf-8')
							# 豆瓣评分定级：1分：很差；2分：较差；3分：有用；4分：推荐；5分：力荐
							pattern = r'<span class="allstar(.*?)0 rating" title=".*?"></span>.*?<p class="">(.*?)<'
							reviews = re.findall(pattern, content, re.S)
							# reviews = [review.replace('<br />', '\n').replace('<br/>', '\n').replace('\n', ' ').strip() for review in reviews]
							for review in reviews:
								score = int(review[0])
								if score == 3:
									continue
								elif score <3:
									label = '0'
								else:
									label = '1'
								real_review = review[1].replace('<br />', '\n').replace('<br/>', '\n').replace('\n', ' ').strip()
								fw.write(real_review.encode('utf8') + '\t'+ label + '\n')
							i += 1
							print '[finished] %.2f%%' % (float(i*100)/page_num)
							time.sleep(0.1)

						except urllib2.URLError, e:
							print '[Error] can\'t get the page %d' % i
							print 'Retrying......'
	else:
		print response.geturl()
		print 'Login failed!'


myscore = {
	u"很差":1,
	u"较差":2,
	u"有用":3,
	u"推荐":4,
	u"力荐":5
}


if __name__ == '__main__':
	# item_num = r'B00JZ96ZI8'
	# getReviewsFromAmazon(6755, 444, item_num)

	url = r'https://movie.douban.com/subject/19944106/comments?limit=20&sort=new_score&start='
	review_number = 125449
	movie_name = 'meirenyu'
	getReviewFromDouban(url, review_number, movie_name)





