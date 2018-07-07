import numpy as np
import os

from os.path import join, isfile
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.decomposition import PCA

import matplotlib.pyplot as pltd
from sklearn.cluster import KMeans

from collections import Counter


path = '/Users/aadil/cb-02_internship/corpus'
title_path = '/Users/aadil/cb-02_internship/titles'
dump_path = '/Users/aadil/cb-02_internship/cluster'
stop_words = set(stopwords.words('english'))


filepaths = [join(path, f) for f in os.listdir(path) if isfile(join(path, f))]
filepaths.sort()

filepaths_t = [join(title_path, f) for f in os.listdir(title_path) if isfile(join(title_path, f))]
filepaths_t.sort()

corpus = [open(f, 'r').read() for f in filepaths]
corpus = np.array(corpus)

titles = [open(f, 'r').read() for f in filepaths_t]
titles = np.array(titles)

ps = PorterStemmer()
for i in range (corpus.shape[0]):
	res = [ps.stem(word.lower()) for word in corpus[i].split() if word.lower() not in stop_words and word.isalpha()]
	corpus[i] = ' '.join(res)



tf = TfidfVectorizer()
transformed = tf.fit_transform(raw_documents=corpus)
transformed = np.array(transformed.todense())



transformed = PCA(n_components=2).fit_transform(transformed)

kmeans = KMeans(n_clusters=4, max_iter=6000)
kmeans.fit(transformed)

centroids = kmeans.cluster_centers_
labels = kmeans.labels_


print (labels)

for i in range(5):
	indices = [index for index, value in enumerate(labels) if value == i]
	for index in indices:
		print (titles[index])
	

	print ()
	print ()

color= ['red' if l == 0 else 'green' if l == 1 else 'pink' if l == 2 else 'yellow' if l == 3 else 'blue' for l in labels]

plt.scatter(transformed[:, 0], transformed[:, 1],  c=color, s=30)
plt.show()


## ----- Dumping data from clusters to different folders. Uncomment only if data has to be redumped according to clusters -------

# corpus = [open(f, 'r').read() for f in filepaths]
# corpus = np.array(corpus)

# print (corpus[0])

# filename = 'doc'
# labels = np.array(labels)
# i = 0
# for i in range(labels.shape[0]):
# 	f = open(os.path.join(dump_path + str(labels[i]), (filename + str(i) +  '.txt')), 'w')
# 	f.write(corpus[i])





