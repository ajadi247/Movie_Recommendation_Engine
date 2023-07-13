# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 22:31:08 2020

@author: ajadi
"""


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

text = ['London Paris London', 'Paris Paris London']
cv = CountVectorizer()
count_matrix = cv.fit_transform(text)

print(cv.get_feature_names())
count_matrix = count_matrix.toarray()
print('count matrix:',count_matrix)
similarity_scores = cosine_similarity(count_matrix)
print('similarity Scores:', similarity_scores)