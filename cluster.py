import numpy as np
import os

from os.path import join, isfile
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

path = '/Users/aadil/cb-02_internship/corpus'
stop_words = set(stopwords.words('english'))


filepaths = [join(path, f) for f in os.listdir(path) if isfile(join(path, f))]
filepaths.sort()

corpus = [open(f, 'r').read() for f in filepaths]
corpus = np.array(corpus)

#print (corpus[0])
print ()

ps = PorterStemmer()
for i in range (corpus.shape[0]):
	res = [ps.stem(word.lower()) for word in corpus[i].split() if word.lower() not in stop_words and word.isalpha()]
	corpus[i] = ' '.join(res)

#print (corpus[0])

tf = TfidfVectorizer()
transformed = tf.fit_transform(raw_documents=corpus)
transformed = np.array(transformed.todense())

transformed = PCA(n_components=2).fit_transform(transformed)
kmeans = KMeans(n_clusters=3, max_iter=600)
kmeans.fit(transformed)

centroids = kmeans.cluster_centers_
labels = kmeans.labels_


# pca = PCA(n_components=2)
# x = pca.fit_transform(transformed)
# centroids = pca.fit_transform(centroids)

print (labels)
print (transformed.shape)

print (corpus[-2])

# a,b = y.T

color= ['red' if l == 0 else 'green' if l == 1 else 'blue' for l in labels]

plt.scatter(transformed[:, 0], transformed[:, 1],  c=color, s=30)
plt.show()



# fig = plt.figure(figsize = (8,8))
# ax = fig.add_subplot(1,1,1)
# targets = [0, 1, 2]
# colors = ['r', 'g', 'c']
# for target,color in zip(targets,colors):
# 	indices = labels[target] == target
# 	ax.scatter(x[indices,0],x[indices,1],c = color, s=50)


# print(indices)
# ax.legend(targets)
# ax.grid()
# plt.show()


# print (tf.vocabulary_)
# print (transformed[0:5])
# print (transformed.shape)

