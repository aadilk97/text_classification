import numpy as np
import os
import nltk

from os.path import join, isfile

from nltk.corpus import stopwords
from nltk.tree import Tree
from nltk.stem import PorterStemmer


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


from collections import Counter

## Finding the entity from a local list of a particlar document
def find_in_list(local_list, target):
	tnames = target.split()
	flag = False
	for entity in local_list:
		e = entity.split()
		for tname in tnames:
			if tname in e:
				flag = True
			else:
				flag = False

		if flag:
			return entity

	return ''

## Finding target entity from a list of known entites and organizations
def find_in_list2(entities, orgs, target):
	tnames = target.split()
	flag = False
	for entity in entities:
		e = entity.split()
		for tname in tnames:
			if tname in e:
				flag = True
			else:
				flag = False

		if flag:
			return entity

	for entity in orgs:
		e = entity.split()
		for tname in tnames:
			if tname in e:
				flag = True
			else:
				flag = False

		if flag:
			return entity

	return ''


path = '/Users/aadil/cb-02_internship/cluster1'
stop_words = set(stopwords.words('english'))


filepaths = [join(path, f) for f in os.listdir(path) if isfile(join(path, f))]
filepaths.sort()

corpus = [open(f, 'r').read() for f in filepaths]
corpus = np.array(corpus)

dump = ''
entities = []
organizations = []
for j in range(corpus.shape[0]):
	dump += corpus[j]


tokenized = nltk.word_tokenize(dump)
tagged = nltk.pos_tag(tokenized)

## Generating a list of all entities
namedEnt = nltk.ne_chunk(tagged)
for i in namedEnt:
	if type(i) == Tree:
		for subtree in i.subtrees():
			name = ''
			for leaf in subtree.leaves():
				leaf_parts = list(leaf[0])
				for part in leaf_parts:
					name += part
				name += ' '

			if subtree.label() == 'PERSON' and len(subtree) > 1:
				
				if name not in entities:
					entities.append(name)

			if subtree.label() == 'ORGANIZATION' and len(subtree) > 1:
				if name not in entities and name not in organizations:
					organizations.append(name)



## Mapping each entity with the one in the list previously generated
for j in range(corpus.shape[0]):
	text = corpus[j]
	new_text = ''

	local_list = []
	tokenized = nltk.word_tokenize(text)
	tagged = nltk.pos_tag(tokenized)
	namedEnt = nltk.ne_chunk(tagged)

	for i in namedEnt:
		if type(i) == Tree:
			for subtree in i.subtrees():
				name = ''
				if subtree.label() == 'PERSON' or subtree.label() == 'ORGANIZATION':
					for leaf in subtree.leaves():
						name += leaf[0] + ' '


					res = find_in_list(local_list, name)
					if res == '':
						res = find_in_list2(entities, organizations, name)
						local_list.append(res)

					for word in res.split():
						new_text += word
				new_text += ' '		

	corpus[j] = new_text

## Finding the most frequently occuring entity in a given document
	counts = Counter(corpus[j].split())
	for word, freq in (counts.most_common(1)):
		print ("i = ", j, word)



## ----- Checking for similarity ------

corpus = [open(f, 'r').read() for f in filepaths]
corpus = np.array(corpus)

ps = PorterStemmer() 
for i in range (corpus.shape[0]):
	res = [ps.stem(word.lower()) for word in corpus[i].split() if word.lower() not in stop_words and word.isalpha()]
	corpus[i] = ' '.join(res)

tf = TfidfVectorizer()
transformed = tf.fit_transform(corpus)
pairwise_similarity = (transformed * transformed.T).A

## Printing out the similarity values for the last document
print (pairwise_similarity[-1])


## ----- End of similarity ------

