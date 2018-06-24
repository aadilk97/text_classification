from goose3 import Goose
from googlesearch import search

import os


# queries = ['trump and kim summit', 'narendra modi up elections', 'rahul gandhi karnataka', 'pranab mukherjee rss', 'steve smith ball tampering', 'sylvester stallone news', 'kylie jenner news',
#  'news on taylor swift', ' news on priyanka chopra', 'salman khan news', 'jio news', 'ab de villiers retirement', 'virat kohli injury, 'facebook data leak', 'facebook ai']

queries = ['roger federer', 'rafael nadal', 'ronaldo', 'google deep mind', 'android']


g = Goose()
path_doc = '/Users/aadil/cb-02_internship/corpus'
path_title = '/Users/aadil/cb-02_internship/titles'

i = 74
for query in queries:
	count = 0
	filename = 'doc'
	print ("Dumping news for ", query)

	for url in search(query, stop=2):
		try:
			article = g.extract(url=url)

		except:
			continue
		#print (article.title)

		if count < 5:
			if len(article.cleaned_text) >= 100:
				f = open(os.path.join(path_doc, (filename + str(i) +  '.txt')), 'w')
				f.write(article.cleaned_text)
				f.close()

				f = open(os.path.join(path_title, (filename + str(i) +  '.txt')), 'w')
				f.write(article.title)
				f.close()

				i += 1
				count += 1

		else:
			break

	print ()



	


        
    

