import multiprocessing
from pickle import dump, load

import numpy as np
import pandas as pd
from lxml import etree

import preprocessing.dictionary_features as pdf
from preprocessing.article_class import ArticleClass


def save_dataset(dataset, filename):
    """
    This method saves a dataset to a file (pickle)
    :param dataset:
    :param filename:
    """
    dump(dataset, open(filename, 'wb'))
    print('Saved: %s' % filename)


def load_dataset(filename):
    """
    This method loads a dataset from a file (pickle)
    :return file: 
    :param filename:
    """
    return load(open(filename, 'rb'))


def read_features(filename):
    """
    This function takes the filename of the feature read its contents and return it to the user
    :param filename: 
    :return: list of features
    """
    features = open("../new_data/dictionary/" + filename + ".csv",
                    encoding="latin-1").readlines()
    new_features = []
    for feature in features:
        new_features.append(feature.replace('\n', ''))

    return new_features


def construct_features_dictionary(features_index):
    """
    This method constructs a feature dictionary.
    :param features_index: 
    :return: a dictionary of features where the keys are feature names and value is a list of all the feature contents
    """
    features = {}
    for feature, index in features_index.items():
        features[feature] = read_features(features_index[feature])
    return features


def text_content(element):
    """
    This method takes an element and extract the text in it. 
    :param element: xml element
    :return: text
    """
    return ' '.join([t.strip() for t in element.itertext()])


class DataProcessing(object):

    def __init__(self):
        self.liwc_features = construct_features_dictionary(pdf.liwc_features_list)
        self.liwc_counts = dict.fromkeys(self.liwc_features.keys(), 0)
        self.liwc_counts_title = dict.fromkeys(self.liwc_features.keys(), 0)
        self.punctuations = dict(question_mark=['?'], exclamation_mark=['!'], quotation_mark=['"'],
                                 paranthesis=['(', ')'], colons=[',', ';', ":"], dot=['.'])
        self.punctuation_counts = dict.fromkeys(self.punctuations.keys(), 0)
        self.punctuation_counts_title = dict.fromkeys(self.punctuations.keys(), 0)
        self.count_articles = 0
        # TODO self.total_hedges = None
        # TODO self.total_opinion_lexicon = None
        # TODO self.total_NRC_VAD_Lexicon = None

    def read_article_text(self, articles_filename):
        """
        This method reads text, title and publication date from articles.
        :param articles_filename:
        """
        context = etree.iterparse(articles_filename, events=('end', 'start'))
        article = None
        for event, elem in context:
            if elem.tag == 'article':
                if event == 'start':
                    article = ArticleClass(self.liwc_features.keys(), self.punctuations.keys())
                else:
                    article.id = elem.attrib['id']
                    article.title = elem.attrib['title']
                    try:
                        article.published_at = elem.attrib['published-at']
                    except KeyError:
                        article.published_at = None
                    yield article
                    article = None

            if article is None:
                continue

            if event == 'end':
                if elem.tag == 'article':
                    article.text = ' '.join([article.text, text_content(elem)])
                    continue
                if elem.tag == 'p':
                    article.text = ' '.join([article.text, text_content(elem)])
                    article.count_paragraphs += 1
                    continue
                if elem.tag == 'q':
                    article.text = ' '.join([article.text, text_content(elem)])
                    article.count_quotes += 1
                    continue
                if elem.tag == 'a':
                    article.text = ' '.join([article.text, text_content(elem)])
                    article.count_urls += 1
                    continue

    def read_articles_label(self, labels_filename):
        """
        This method reads article label and url source
        :param labels_filename:
        """
        context = etree.iterparse(labels_filename, events=('end', 'start'))
        article = None
        for event, elem in context:
            if elem.tag == 'article':
                if event == 'start':
                    article = ArticleClass(self.liwc_features.keys(), self.punctuations.keys())
                else:
                    article.id = elem.attrib['id']
                    article.url = elem.attrib['url']
                    article.hyperpartisan = (1 if elem.attrib['hyperpartisan'] == "true" else 0)
                    yield article
                    article = None

            if article is None:
                continue

    def construct_article_array(self, articles_filename, labels_filename):
        """
        This method construct an article array and counts LIWC and punctuation features in it.
        :param articles_filename:
        :param labels_filename:
        :return: list of articles
        """
        articles = {}

        for article in self.read_article_text(articles_filename):
            articles[article.id] = article

        for article in self.read_articles_label(labels_filename):
            articles[article.id].hyperpartisan = article.hyperpartisan
            articles[article.id].url = article.url
            articles[article.id].bias = article.bias
            articles[article.id].labeled_by = article.labeled_by
            for feature, words in self.liwc_features.items():
                liwc_in_article = 0
                liwc_in_title = 0
                for word in words:
                    counts_articles = articles[article.id].text.count(word)
                    counts_title = articles[article.id].title.count(word)
                    liwc_in_article += counts_articles
                    liwc_in_title += counts_title
                    articles[article.id].liwc_counts[feature] += counts_articles
                    articles[article.id].liwc_counts_title[feature] += counts_title
                    articles[article.id].all_liwc += counts_articles
                    articles[article.id].all_liwc_title += counts_title
                if liwc_in_article >= 1:
                    self.liwc_counts[feature] += 1
                if liwc_in_article >= 1:
                    self.liwc_counts_title[feature] += 1

            for feature, words in self.punctuations.items():
                punc_in_article = 0
                punc_in_title = 0
                for word in words:
                    counts_articles = articles[article.id].text.count(word)
                    counts_title = articles[article.id].title.count(word)
                    punc_in_article += counts_articles
                    punc_in_title += counts_title
                    articles[article.id].punctuation_counts[feature] += counts_articles
                    articles[article.id].punctuation_counts_title[feature] += counts_title
                    articles[article.id].all_punc += counts_articles
                    articles[article.id].all_punc_title += counts_title
                if punc_in_article >= 1:
                    self.punctuation_counts[feature] += 1
                if punc_in_title >= 1:
                    self.punctuation_counts_title[feature] += 1

        self.count_articles = len(articles)
        return articles


def process_data(articles_training, dataprocessor, dataset):
    """
    This method extracts and saves individual features in their own files.
    :param articles_training:
    :param dataprocessor:
    :param dataset:
    :return: training idf of tfidf features
    """
    x_train_unigrams = []
    x_train_titles = []
    x_train_liwc = []
    x_train_punctuation = []
    x_train_liwc_title = []
    x_train_punctuation_title = []
    x_train_structure = []
    id_train = []
    y_train = []

    # Calculate IDF for features
    idf_liwc_article = {}
    idf_punc_article = {}
    idf_liwc_title = {}
    idf_punc_title = {}

    for feature in dataprocessor.liwc_features.keys():
        idf_liwc_article[feature] = 0
        idf_liwc_title[feature] = 0
        if dataprocessor.count_articles != 0:
            idf_liwc_article[feature] = dataprocessor.liwc_counts[feature] / dataprocessor.count_articles
            idf_liwc_title[feature] = dataprocessor.liwc_counts_title[feature] / dataprocessor.count_articles

    for feature in dataprocessor.punctuations.keys():
        idf_punc_article[feature] = 0
        idf_punc_title[feature] = 0
        if dataprocessor.count_articles != 0:
            idf_punc_article[feature] = dataprocessor.punctuation_counts[feature] / dataprocessor.count_articles
            idf_punc_title[feature] = dataprocessor.punctuation_counts_title[feature] / dataprocessor.count_articles

    for article_id in articles_training:
        id_train.append(article_id)
        # Extract unigram features
        x_train_unigrams.append(articles_training[article_id].clean_article())
        # Extract title features
        x_train_titles.append(articles_training[article_id].title)
        # Extract structural features
        x_train_structure.append([articles_training[article_id].count_quotes,
                                  articles_training[article_id].count_paragraphs,
                                  articles_training[article_id].count_urls])
        # Extract liwc features
        liwc = []
        liwc_title = []
        liwc_order = []
        for feature in dataprocessor.liwc_features.keys():
            tf_article = 0
            tf_title = 0
            if articles_training[article_id].all_liwc != 0:
                tf_article = articles_training[article_id].liwc_counts[feature] / articles_training[article_id].all_liwc
            if articles_training[article_id].all_liwc_title != 0:
                tf_title = articles_training[article_id].liwc_counts_title[feature] / articles_training[
                    article_id].all_liwc_title

            liwc.append(tf_article * idf_liwc_article[feature])
            liwc_title.append(tf_title * idf_liwc_title[feature])
            liwc_order.append(feature)

        x_train_liwc.append(liwc)
        x_train_liwc_title.append(liwc_title)

        # Extract punctuation features
        punc = []
        punc_title = []
        punc_order = []
        for feature in dataprocessor.punctuations.keys():
            tf_article = 0
            tf_title = 0
            if articles_training[article_id].all_punc != 0:
                tf_article = articles_training[article_id].punctuation_counts[feature] / articles_training[
                    article_id].all_punc
            if articles_training[article_id].all_punc_title != 0:
                tf_title = articles_training[article_id].punctuation_counts_title[feature] / articles_training[
                    article_id].all_punc_title
            punc.append(tf_article * idf_punc_article[feature])
            punc_title.append(tf_title * idf_punc_title[feature])
            punc_order.append(feature)
        x_train_punctuation.append(punc)
        x_train_punctuation_title.append(punc_title)

        y_train.append(articles_training[article_id].hyperpartisan)

    # Save and export extracted features to be used in classification
    save_dataset([id_train, x_train_unigrams, y_train],
                 '../new_data/pkl-objects/train_unigrams_' + dataset + '.pkl')
    save_dataset([id_train, x_train_structure, y_train],
                 '../new_data/pkl-objects/train_structure_' + dataset + '.pkl')
    save_dataset([id_train, x_train_liwc, y_train],
                 '../new_data/pkl-objects/train_liwc_' + dataset + '.pkl')
    save_dataset([id_train, x_train_punctuation, y_train],
                 '../new_data/pkl-objects/train_punctuation_' + dataset + '.pkl')

    save_dataset([id_train, x_train_titles, y_train],
                 '../new_data/pkl-objects/train_titles_' + dataset + '.pkl')
    save_dataset([id_train, x_train_liwc_title, y_train],
                 '../new_data/pkl-objects/train_liwc_title_' + dataset + '.pkl')
    save_dataset([id_train, x_train_punctuation_title, y_train],
                 '../new_data/pkl-objects/train_punctuation_title_' + dataset + '.pkl')

    # Save ordered feature names for liwc and punctuation
    save_dataset(
        [[key for key in dataprocessor.liwc_features.keys()], [key for key in dataprocessor.punctuations.keys()]],
        "../new_data/pkl-objects/features_order_" + dataset + ".pkl")

    return idf_liwc_article, idf_punc_article, idf_liwc_title, idf_punc_title


data = pd.read_csv(
    '../new_data/dictionary/NRC-Emotion-Lexicon/NRC-Emotion-Lexicon-v0.92/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt',
    sep="	", header=None)
data.columns = ["word", "emotion", "value"]
print("Loaded Emotions")


def map_article_emotions(article):
    """
    This method counts emotions in each given article
    :param article:
    :return: count of emotions in article
    """
    article_emotions = np.zeros((10,), dtype=np.int)
    for word in article:
        word_emotions = list(data.loc[data['word'] == word]["value"])
        if len(word_emotions) != 0:
            article_emotions = np.sum([article_emotions, word_emotions], axis=0)
    return article_emotions


def extract_emotions(articles):
    """
    Extract and count emotions from a list of articles
    :param articles:
    :return: list of extracted counts of emotions from given articles
    """
    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(cores)
    emotions = list(pool.map(map_article_emotions, articles))
    return emotions


if __name__ == '__main__':
    dataprocessor1 = DataProcessing()
    # Extract all information from xml and construct articles array
    articles_training_1 = dataprocessor1.construct_article_array(
        "/tmp/hyperpartisan_project/project/new_data/articles-training-byarticle-20181122.xml",
        "/tmp/hyperpartisan_project/project/new_data/ground-truth-training-byarticle-20181122.xml")

    # Calculate tfidf for all features
    idf_liwc_article1, idf_punc_article1, idf_liwc_title1, idf_punc_title1 = process_data(articles_training_1,
                                                                                          dataprocessor1, "article")

    # Save training idfs for later use in testing
    save_dataset([idf_liwc_article1, idf_punc_article1, idf_liwc_title1, idf_punc_title1],
                 '/tmp/hyperpartisan_project/project/new_data/pkl-objects/train_idf_by_articles.pkl')
