
from __future__ import division

import os
import getopt
import sys
import random
from joblib import dump, load
import pickle as pkl
from scipy.sparse import hstack
from lxml import etree
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from string import punctuation
from typing import List, Any


class ArticleClass:
    def __init__(self, liwc_dict, punc_dict):
        self.id = None
        self.hyperpartisan = "hyperpartisan"
        self.bias = "bias"
        self.title = "title"
        self.text = ""
        self.labeled_by = "labeled_by"
        self.published_at = "published_at"
        self.count_urls = 0
        self.count_paragraphs = 0
        self.count_quotes = 0
        self.hedges = []
        self.boosters = []
        self.negatives = []
        self.positives = []
        self.liwc_counts = dict.fromkeys(liwc_dict, 0)
        self.punctuation_counts = dict.fromkeys(punc_dict, 0)
        self.liwc_counts_title = dict.fromkeys(liwc_dict, 0)
        self.punctuation_counts_title = dict.fromkeys(punc_dict, 0)
        self.all_liwc = 0
        self.all_punc = 0
        self.all_liwc_title = 0
        self.all_punc_title = 0

    def clean_article(self):
        """
        This method cleans the article and stores its text as an array of tokens.
        """
        # split into tokens by white space
        tokens = self.text.split(" ")
        # remove punctuation from each token
        table = str.maketrans('', '', punctuation)
        tokens = [w.translate(table) for w in tokens]  # type: List[Any]
        # remove remaining tokens that are not alphabetic
        tokens = [word for word in tokens if word.isalpha()]
        # filter out stop words
        stop_words = set(stopwords.words('english'))
        tokens = [w for w in tokens if not w in stop_words]
        # lemmatization and lowercase
        lmtzr = WordNetLemmatizer()
        tokens = [lmtzr.lemmatize(w.lower()) for w in tokens]
        # filter out short tokens
        tokens = [word for word in tokens if len(word) > 1]
        return tokens


liwc_features_list = dict(Funct="1", Pronoun="2", Ppron="3", I="4", We="5", You="6", SheHe="7", They="8", Ipron="9",
                          Article="10", Verbs="11", AuxVb="12", Past="13", Present="14", Future="15", Adverbs="16",
                          Prep="17", Conj="18", Negate="19", Quant="20", Numbers="21", Swear="22", Social="23",
                          Family="24", Friends="25", Humans="26", Affect="27", Posemo="28", Negemo="29", Anx="30",
                          Anger="30", Sad="31", CogMech="32", Insight="33", Cause="34", Discrep="35", Tentat="36",
                          Certain="37", Inhib="38", Incl="39", Excl="40", Percept="41", See="42", Hear="43", Feel="44",
                          Bio="45", Body="46", Health="47", Sexual="48", Ingest="49", Relativ="50", Motion="51",
                          Space="52", Time="53", Work="54", Achiev="55", Leisure="56", Home="57", Money="58", Relig="59",
                          Death="60", Assent="61", Nonflu="62", Filler="63")


random.seed(42)
runOutputFileName = "prediction.txt"


def load_dataset(filename):
    return pkl.load(open(filename, 'rb'))


def parse_options():
    """Parses the command line options."""
    try:
        long_options = ["inputDataset=", "outputDir="]
        opts, _ = getopt.getopt(sys.argv[1:], "d:o:", long_options)
    except getopt.GetoptError as err:
        print(str(err))
        sys.exit(2)

    inputDataset = "undefined"
    outputDir = "undefined"

    for opt, arg in opts:
        if opt in ("-d", "--inputDataset"):
            inputDataset = arg
        elif opt in ("-o", "--outputDir"):
            outputDir = arg
        else:
            assert False, "Unknown option."
    if inputDataset == "undefined":
        sys.exit("Input dataset, the directory that contains the articles XML file, is undefined. Use option -d or --inputDataset.")
    elif not os.path.exists(inputDataset):
        sys.exit("The input dataset folder does not exist (%s)." % inputDataset)

    if outputDir == "undefined":
        sys.exit("Output path, the directory into which the predictions should be written, is undefined. Use option -o or --outputDir.")
    elif not os.path.exists(outputDir):
        os.mkdir(outputDir)

    return (inputDataset, outputDir)


########## LXML ##########


def count_features(filename):
    features = open("dictionary/" + filename + ".csv",
                    encoding="latin-1").readlines()
    new_features = []
    for feature in features:
        new_features.append(feature.replace('\n', ''))

    return new_features


def construct_features_dictionary(features_index):
    features = {}
    for feature, index in features_index.items():
        features[feature] = count_features(features_index[feature])
    return features


class DataProcessing(object):
    def __init__(self, outFile):
        self.outFile = outFile
        self.liwc_features = construct_features_dictionary(liwc_features_list)
        self.punctuations = dict(question_mark=['?'], exclamation_mark=['!'], quotation_mark=['"'],
                                 paranthesis=['(', ')'], colons=[',', ';', ":"], dot=['.'])

    def text_content(self, elt):
        return ' '.join([t.strip() for t in elt.itertext()])

    def efficient_read_article_text(self, articles_filename):
        context = etree.iterparse(articles_filename, events=('end', 'start'))
        article = None
        for event, elem in context:
            if elem.tag == 'article':
                if event == 'start':
                    article = ArticleClass(self.liwc_features.keys(), self.punctuations.keys())
                else:
                    article.id = elem.attrib['id']
                    article.title = elem.attrib['title']
                    yield article
                    article = None

            if article is None:
                continue

            if event == 'end':
                if elem.tag == 'article':
                    article.text = ' '.join([article.text, self.text_content(elem)])
                    continue
                if elem.tag == 'p':
                    article.text = ' '.join([article.text, self.text_content(elem)])
                    article.count_paragraphs += 1
                    continue
                if elem.tag == 'q':
                    article.text = ' '.join([article.text, self.text_content(elem)])
                    article.count_quotes += 1
                    continue
                if elem.tag == 'a':
                    article.text = ' '.join([article.text, self.text_content(elem)])
                    article.count_urls += 1
                    continue

    def classify_article(self, articles_filename):
        # Load Classifier:
        idf_liwc_article, idf_punc_article, idf_liwc_title, idf_punc_title = load_dataset("train_idf_by_articles.pkl")
        classifier = load("byarticle_classification_model.pkl")
        svd = load("byarticle_svd_model.pkl")
        tfidf = load("byarticle_tfidf_model.pkl")

        for article in self.efficient_read_article_text(articles_filename):
            for feature, words in self.liwc_features.items():
                liwc_in_article = 0
                liwc_in_title = 0
                for word in words:
                    counts_articles = article.text.count(word)
                    counts_title = article.title.count(word)
                    liwc_in_article += counts_articles
                    liwc_in_title += counts_title
                    article.liwc_counts[feature] += counts_articles
                    article.liwc_counts_title[feature] += counts_title
                    article.all_liwc += counts_articles
                    article.all_liwc_title += counts_title

            for feature, words in self.punctuations.items():
                punc_in_article = 0
                punc_in_title = 0
                for word in words:
                    counts_articles = article.text.count(word)
                    counts_title = article.title.count(word)
                    punc_in_article += counts_articles
                    punc_in_title += counts_title
                    article.punctuation_counts[feature] += counts_articles
                    article.punctuation_counts_title[feature] += counts_title
                    article.all_punc += counts_articles
                    article.all_punc_title += counts_title
            # Prepare features
            unigrams = tfidf.transform([" ".join(article.clean_article())])
            liwc = []
            punctuation = []
            structure = [article.count_quotes,
                                     article.count_paragraphs,
                                     article.count_urls]

            for feature in self.liwc_features.keys():
                tf_article = 0
                if article.all_liwc != 0:
                    tf_article = article.liwc_counts[feature] /article.all_liwc
                liwc.append(tf_article * idf_liwc_article[feature])

            for feature in self.punctuations.keys():
                tf_article = 0
                if article.all_punc != 0:
                    tf_article = article.punctuation_counts[feature] / article.all_punc
                if article.all_punc_title != 0:
                    tf_title = article.punctuation_counts_title[feature] / article.all_punc_title
                punctuation.append(tf_article * idf_punc_article[feature])

            test_article = hstack([unigrams, [liwc], [punctuation], [structure]])
            test_article = svd.transform(test_article)
            # Classify Article
            clf_pred = classifier.predict(test_article)[0]
            prediction = ("true" if clf_pred == 1 else "false")
            confidence = max(classifier.predict_proba(test_article)[0])
            # Output prediction
            self.outFile.write(article.id + " " + prediction + " " + str(confidence) + "\n")


########## MAIN ##########


def main(inputDataset, outputDir):
    """Main method of this module."""

    with open(outputDir + "/" + runOutputFileName, 'w') as outFile:
        dataprocessor = DataProcessing(outFile)
        for file in os.listdir(inputDataset):
            if file.endswith(".xml"):
                inputRunFile = inputDataset + "/" + file
                dataprocessor.classify_article(inputRunFile)



    print("The predictions have been written to the output folder.")


if __name__ == '__main__':
    main(*parse_options())

