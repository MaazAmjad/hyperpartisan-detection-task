# -"- coding: utf-8 -"-
from __future__ import division
import re
from gensim import corpora, models, similarities
import pickle
import numpy as np
import pandas as pd
import os
import operator
import nltk
import csv

# os.getcwd()


def print_topics(lda_model,n_topics,n_words=15):
    print('\n============================')
    print('printing the top {} words in {} topics'.format(n_words,n_topics))
    for topic in lda_model.show_topics(n_topics, n_words):
        words = ""
        for i, word in enumerate(topic[1].split('+')):
            if i == len(topic[1].split('+')) -1:
                words += " "+word.split('*')[1][1:-1]
            else:
                words += " "+word.split('*')[1][1:-2]
        print(words+"\n")
    print('============================\n')


stopwords = set(nltk.corpus.stopwords.words('english'))

path = "/home/tariq/Downloads/datasets/hyperpartisan/"
train_data, label_train = pickle.load(open(path+"train_v.pkl", 'rb'))
test_data, label_test = pickle.load(open(path+"test_v.pkl", 'rb'))

# Vectorizing and dictionariy creation
texts = [[token for token in text if token not in stopwords] for text in train_data]
test_texts = [[token for token in text if token not in stopwords] for text in test_data]
print('Done with texts')
dictionary = corpora.Dictionary(texts)
print('Done with dictionary')
corpus = [dictionary.doc2bow(text) for text in texts]
print('Done with training corpus')
test_corpus = [dictionary.doc2bow(text) for text in test_texts]
print('Done with test corpus')

#--------------------------------------------------------------------------
# LDA with 5 topics
lda5 = models.LdaModel(corpus, id2word=dictionary, num_topics=5)
lda5.save('model/lda5.model')
print('Done with LDA 5')
corpus_lda5 = lda5[corpus]
test_lda5 = lda5[test_corpus]
print('Done with topic distribution of training data under LDA 5')
print_topics(lda5,5)
# ============================
# printing the top 15 words in 5 topics
#  said state trump would president republican law people new say year house one court also
#  said one say like people year time new first get two know game day dont
#  de la el en que los del un film una con por se para su
#  war state government country people said would one american world military president united political also
#  company year said percent million market stock new billion share business price also bank would
# ============================

with open('document2topic/training5topics.csv', 'w',newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for line in corpus_lda5:
            writer.writerow(line)
with open('document2topic/test5topics.csv', 'w',newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for line in test_lda5:
            writer.writerow(line)



#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
# LDA with 10 topics
# lda10 = models.LdaModel(corpus, id2word=dictionary, num_topics=10)
lda10 =  models.LdaModel.load('model/lda.model')
print('Done with LDA 10')
corpus_lda10 = lda10[corpus]
test_lda10 = lda10[test_corpus]
print('Done with topic distribution of training data under LDA 10')
print_topics(lda10,10)

with open('document2topic/training10topics.csv', 'w',newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for line in corpus_lda10:
            writer.writerow(line)

with open('document2topic/test10topics.csv', 'w',newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for line in test_lda10:
            writer.writerow(line)

#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
# LDA with 30 topics, same number as the NYT dataset
lda30 = models.LdaModel(corpus, id2word=dictionary, num_topics=30)
lda30.save('model/lda30.model')
print('Done with LDA 30')
corpus_lda30 = lda30[corpus]
test_lda30 = lda30[test_corpus]
print('Done with topic distribution of training data under LDA 30')
print_topics(lda30,30)

with open('document2topic/training30topics.csv', 'w',newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for line in corpus_lda30:
            writer.writerow(line)
with open('document2topic/test30topics.csv', 'w',newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for line in test_lda30:
            writer.writerow(line)
#--------------------------------------------------------------------------




#----------------------------Model Evaluation------------------------------
from gensim.models import CoherenceModel
print('\nPerplexity: ', lda5.log_perplexity(corpus))
coherence_model_lda5 = CoherenceModel(model=lda5, texts=texts, dictionary=dictionary, coherence='c_v')
coherence_lda5 = coherence_model_lda5.get_coherence()
print('\nCoherence Score: ', coherence_lda5)
# Perplexity:  -8.709969881488803
# 

print('\nPerplexity: ', lda10.log_perplexity(corpus))
coherence_model_lda10 = CoherenceModel(model=lda10, texts=texts, dictionary=dictionary, coherence='c_v')
coherence_lda10 = coherence_model_lda10.get_coherence()
print('\nCoherence Score: ', coherence_lda10)

print('\nPerplexity: ', lda30.log_perplexity(corpus))
coherence_model_lda30 = CoherenceModel(model=lda30, texts=texts, dictionary=dictionary, coherence='c_v')
coherence_lda30 = coherence_model_lda30.get_coherence()
print('\nCoherence Score: ', coherence_lda30)
#----------------------------------------------------------------------------