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
# ============================
# printing the top 15 words in 10 topics
#  state year new would percent worker government tax water job people million program health economic
#  said game first year one team time two point new season last like get play
#  said state court law case department new say official federal information county report attorney also
#  de la el en que los del un una por se con para su al
#  war military state government country president israel iraq united attack said force security would russia
#  trump president republican said obama house would campaign election state clinton party democrat vote bill
#  people one like right would know even think way american time dont thing make many
#  said school woman police say student child people family year one city church day home
#  company year stock business million share market sale investor fool one billion revenue also new
#  said percent china reuters market bank year trade price company last billion week new oil
# ============================
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
# ============================
# printing the top 15 words in 30 topics
#  said say county prison advertisement year two charge albuquerque ap attorney according police found state
#  trump president white said donald news clinton obama medium hillary fox house american people comment
#  de la el en que los del un una por se con para su al
#  tax year percent job would health rate worker government plan budget cost pay cut care
#  people like think say dont get one know thing thats going want way make would
#  league team san club game world first run year player said hit cup two baseball
#  said syria attack government syrian group muslim state country islamic saudi force military refugee israel
#  point second game first scored said lead win shot two play half goal minute ap
#  energy oil climate gas year plant company industry car power change vehicle global environmental cost
#  school student university education teacher college program community year high say new child chicago work
#  political people world state israel american right one palestinian movement country government power many social
#  day one family said year like time life home back woman child would friend say
#  film show movie star food new also year award actor best hollywood series restaurant director
#  said china reuters trade would north korea chinese country percent united european standard reporting state
#  drug health medical people study patient child said immigrant care immigration doctor year marijuana hospital
#  russian report information russia investigation official intelligence news medium email government security former time fbi
#  bank financial company money million business fund credit loan debt new interest investment billion home
#  law court right case state justice legal would government judge federal supreme act decision order
#  million flight two three ap airline estimated four one number plane five evening pick jackpot
#  said state would new bill city california house year federal department member official committee plan
#  stock year share market percent company investor price growth fool billion sale quarter million motley
#  mexico mexican cuba country brazil spanish cuban latin venezuela chavez spain government castro president de
#  war military iraq american president iran bush state nuclear would force united administration weapon obama
#  book new one story music time world song work art like york first history life
#  game said team season player coach year last play first week sport new yard football
#  police gun officer people said city shooting killed violence one shot protest video man black
#  woman church baptist said christian sexual god gay religious marriage abortion catholic men sex life
#  company business new service apple store customer technology facebook product online said also user year
#  water city area park land river south new north animal island storm national west mile
#  republican party election democrat campaign vote candidate voter obama democratic president political senate clinton house
# ============================
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
# Perplexity:  -8.627242209334147
# 

print('\nPerplexity: ', lda30.log_perplexity(corpus))
coherence_model_lda30 = CoherenceModel(model=lda30, texts=texts, dictionary=dictionary, coherence='c_v')
coherence_lda30 = coherence_model_lda30.get_coherence()
print('\nCoherence Score: ', coherence_lda30)
# Perplexity:  -8.586065863048542
#
#----------------------------------------------------------------------------