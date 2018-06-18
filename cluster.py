import numpy as np
import os

from os.path import join, isfile
from sklearn.feature_extraction.text import TfidfVectorizer

path = '/Users/aadil/cb-02_internship/corpus'

filepaths = [join(path, f) for f in os.listdir(path) if isfile(join(path, f))]
corpus = [open(f, 'r').read() for f in filepaths]

tf = TfidfVectorizer()
transformed = tf.fit_transform(raw_documents=corpus)

# print (tf.vocabulary_)
transformed = np.array(transformed.todense())

print (tf.vocabulary_)
print (transformed[0:5])
print (transformed.shape)

