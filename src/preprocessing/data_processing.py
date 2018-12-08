from pickle import dump, load
from xml.dom import minidom

from lxml import etree

import preprocessing.dictionary_features as pdf
from preprocessing.article_class import ArticleClass


# from sklearn.feature_extraction.text import TfidfVectorizer


# save a dataset to file
def save_dataset(dataset, filename):
    dump(dataset, open(filename, 'wb'))
    print('Saved: %s' % filename)


# load a clean dataset
def load_dataset(filename):
    return load(open(filename, 'rb'))


def count_features(filename):
    features = open("/tmp/hyperpartisan_project/project/new_data/dictionary/" + filename + ".csv",
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
    def __init__(self):
        self.liwc_features = construct_features_dictionary(pdf.liwc_features_list)
        self.liwc_counts = dict.fromkeys(self.liwc_features.keys(), 0)
        self.liwc_counts_title = dict.fromkeys(self.liwc_features.keys(), 0)
        self.punctuations = dict(question_mark=['?'], exclamation_mark=['!'], quotation_mark=['"'],
                                 paranthesis=['(', ')'], colons=[',', ';', ":"], dot=['.'])
        self.punctuation_counts = dict.fromkeys(self.punctuations.keys(), 0)
        self.punctuation_counts_title = dict.fromkeys(self.punctuations.keys(), 0)

        # self.total_hedges = None
        # self.total_opinion_lexicon = None
        # self.total_NRC_VAD_Lexicon = None
        self.count_articles = 0

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
                    try:
                        article.published_at = elem.attrib['published-at']
                    except KeyError:
                        article.published_at = None
                    yield article
                    article = None

            if article == None:
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

    def efficient_read_articles_label(self, labels_filename):
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
                    # article.bias = elem.attrib['bias']
                    article.labeled_by = elem.attrib['labeled-by']
                    yield article
                    article = None

            if article == None:
                continue

    def construct_article_array(self, articles_filename, labels_filename):
        articles = {}
        for article in self.efficient_read_article_text(articles_filename):
            articles[article.id] = article

        for article in self.efficient_read_articles_label(labels_filename):
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

    def old_read_articles(self, articles_filename, labels_filename, training=True):
        # parse an xml file by name
        # if training:
        #    vocabulary = []
        articles_file = minidom.parse(articles_filename)
        labels_file = minidom.parse(labels_filename)
        articles_data = articles_file.getElementsByTagName('article')
        articles_labels = labels_file.getElementsByTagName('article')
        articles = dict()
        for article in articles_data:
            article_id = article.attributes['id'].value
            articles[article_id] = ArticleClass(article_id)
            try:
                articles[article_id].published_at = article.attributes['published-at'].value
            except KeyError:
                articles[article_id].published_at = None
            articles[article_id].title = article.attributes['title'].value
            articles[article_id].text = ""
            paragraphs = article.childNodes
            for paragraph in paragraphs:
                if paragraph.nodeType == paragraph.TEXT_NODE:
                    articles[article_id].text += paragraph.data
                articles[article_id].text += " ".join(p.data for p in paragraph.childNodes if p.nodeType == p.TEXT_NODE)

            articles[article_id].clean_article()
            # articles[article_id].split_to_sentences()
            # if training:
            #    vocabulary.extend(articles[article_id].text)
        for label in articles_labels:
            article_id = label.attributes['id'].value
            articles[article_id].hyperpartisan = (1 if label.attributes['hyperpartisan'].value == "true" else 0)
            # articles[article_id].bias = label.attributes['bias'].value
            articles[article_id].labeled_by = label.attributes['labeled-by'].value
        print("Done reading data")
        # if training:
        #    self.vectorizer = self.extract_features(vocabulary)
        #    print("Done extracting features")

        return articles

    def old_read_titles(self, articles_filename, labels_filename):
        articles_file = minidom.parse(articles_filename)
        labels_file = minidom.parse(labels_filename)
        articles_data = articles_file.getElementsByTagName('article')
        articles_labels = labels_file.getElementsByTagName('article')
        articles = dict()
        for article in articles_data:
            article_id = article.attributes['id'].value
            articles[article_id] = ArticleClass(article_id)
            articles[article_id].title = article.attributes['title'].value
            articles[article_id].clean_title()

        for label in articles_labels:
            article_id = label.attributes['id'].value
            articles[article_id].hyperpartisan = (1 if label.attributes['hyperpartisan'].value == "true" else 0)
        print("Done reading data")

        return articles


if __name__ == '__main__':
    dataprocessor = DataProcessing()
    articles_training = dataprocessor.construct_article_array(
        "/tmp/hyperpartisan_project/project/new_data/articles-training-byarticle-20181122.xml",
        "/tmp/hyperpartisan_project/project/new_data/ground-truth-training-byarticle-20181122.xml")

    articles_training_2 = dataprocessor.construct_article_array(
        "/tmp/hyperpartisan_project/project/new_data/articles-training-bypublisher-20181122.xml",
        "/tmp/hyperpartisan_project/project/new_data/ground-truth-training-bypublisher-20181122.xml")

    articles_training.update(articles_training_2)

    X_train_unigrams = []
    X_train_titles = []
    X_train_liwc = []
    X_train_punctuation = []
    X_train_liwc_title = []
    X_train_punctuation_title = []
    X_train_structure = []
    id_train = []
    y_train = []

    # Calculate IDF for features
    idf_liwc_article = {}
    idf_punc_article = {}
    idf_liwc_title = {}
    idf_punc_title = {}

    # idf_punc_title =  {'colons': 0.32878833333333335, 'dot': 0.09152333333333333, 'exclamation_mark': 0.011135,
    # 'paranthesis': 0.018361666666666665, 'question_mark': 0.05642666666666667,
    # 'quotation_mark': 0.0067283333333333336}

    # idf_punc_article = {'colons': 0.995825,
    #  'dot': 0.9955633333333334,
    #  'exclamation_mark': 0.14768833333333334,
    #  'paranthesis': 0.6076216666666666,
    #  'question_mark': 0.31374833333333335,
    #  'quotation_mark': 0.28462}

    # idf_liwc_title {'Achiev': 0.982685,
    #  'Adverbs': 0.9875566666666666,
    #  'Affect': 0.9953533333333333,
    #  'Anger': 0.56981,
    #  'Anx': 0.56981,
    #  'Article': 0.9999333333333333,
    #  'Assent': 0.87176,
    #  'AuxVb': 0.99932,
    #  'Bio': 0.8658166666666667,
    #  'Body': 0.998315,
    #  'Cause': 0.91621,
    #  'Certain': 0.99417,
    #  'CogMech': 0.85734,
    #  'Conj': 0.9970966666666666,
    #  'Death': 0.817125,
    #  'Discrep': 0.9555216666666667,
    #  'Excl': 0.996455,
    #  'Family': 0.9895116666666667,
    #  'Feel': 0.9511183333333333,
    #  'Filler': 0.9993566666666667,
    #  'Friends': 0.5006066666666666,
    #  'Funct': 0.9999616666666666,
    #  'Future': 0.817915,
    #  'Health': 0.9641183333333333,
    #  'Hear': 0.9214833333333333,
    #  'Home': 0.97386,
    #  'Humans': 0.93519,
    #  'I': 0.99993,
    #  'Incl': 0.8295683333333334,
    #  'Ingest': 0.9643466666666667,
    #  'Inhib': 0.931445,
    #  'Insight': 0.9996266666666667,
    #  'Ipron': 0.9851883333333333,
    #  'Leisure': 0.9797633333333333,
    #  'Money': 0.8811416666666667,
    #  'Motion': 0.9997266666666667,
    #  'Negate': 0.92342,
    #  'Negemo': 0.9822966666666667,
    #  'Nonflu': 0.9945633333333334,
    #  'Numbers': 0.949305,
    #  'Past': 0.99292,
    #  'Percept': 0.9932416666666667,
    #  'Posemo': 0.9929416666666666,
    #  'Ppron': 0.9999333333333333,
    #  'Prep': 0.999735,
    #  'Present': 0.999225,
    #  'Pronoun': 0.9999333333333333,
    #  'Quant': 0.9867116666666667,
    #  'Relativ': 0.9513383333333333,
    #  'Relig': 0.9438666666666666,
    #  'Sad': 0.8994066666666667,
    #  'See': 0.9838666666666667,
    #  'Sexual': 0.978115,
    #  'SheHe': 0.9977666666666667,
    #  'Social': 0.999875,
    #  'Space': 0.985215,
    #  'Swear': 0.7030516666666666,
    #  'Tentat': 0.9016033333333333,
    #  'They': 0.727005,
    #  'Time': 0.9996983333333334,
    #  'Verbs': 0.9995933333333333,
    #  'We': 0.9853266666666667,
    #  'Work': 0.9972916666666667,
    #  'You': 0.8662666666666666}

    #  idf_liwc_article {'Achiev': 0.982685,
    #  'Adverbs': 0.9875566666666666,
    #  'Affect': 0.9953533333333333,
    #  'Anger': 0.56981,
    #  'Anx': 0.56981,
    #  'Article': 0.9999333333333333,
    #  'Assent': 0.87176,
    #  'AuxVb': 0.99932,
    #  'Bio': 0.8658166666666667,
    #  'Body': 0.998315,
    #  'Cause': 0.91621,
    #  'Certain': 0.99417,
    #  'CogMech': 0.85734,
    #  'Conj': 0.9970966666666666,
    #  'Death': 0.817125,
    #  'Discrep': 0.9555216666666667,
    #  'Excl': 0.996455,
    #  'Family': 0.9895116666666667,
    #  'Feel': 0.9511183333333333,
    #  'Filler': 0.9993566666666667,
    #  'Friends': 0.5006066666666666,
    #  'Funct': 0.9999616666666666,
    #  'Future': 0.817915,
    #  'Health': 0.9641183333333333,
    #  'Hear': 0.9214833333333333,
    #  'Home': 0.97386,
    #  'Humans': 0.93519,
    #  'I': 0.99993,
    #  'Incl': 0.8295683333333334,
    #  'Ingest': 0.9643466666666667,
    #  'Inhib': 0.931445,
    #  'Insight': 0.9996266666666667,
    #  'Ipron': 0.9851883333333333,
    #  'Leisure': 0.9797633333333333,
    #  'Money': 0.8811416666666667,
    #  'Motion': 0.9997266666666667,
    #  'Negate': 0.92342,
    #  'Negemo': 0.9822966666666667,
    #  'Nonflu': 0.9945633333333334,
    #  'Numbers': 0.949305,
    #  'Past': 0.99292,
    #  'Percept': 0.9932416666666667,
    #  'Posemo': 0.9929416666666666,
    #  'Ppron': 0.9999333333333333,
    #  'Prep': 0.999735,
    #  'Present': 0.999225,
    #  'Pronoun': 0.9999333333333333,
    #  'Quant': 0.9867116666666667,
    #  'Relativ': 0.9513383333333333,
    #  'Relig': 0.9438666666666666,
    #  'Sad': 0.8994066666666667,
    #  'See': 0.9838666666666667,
    #  'Sexual': 0.978115,
    #  'SheHe': 0.9977666666666667,
    #  'Social': 0.999875,
    #  'Space': 0.985215,
    #  'Swear': 0.7030516666666666,
    #  'Tentat': 0.9016033333333333,
    #  'They': 0.727005,
    #  'Time': 0.9996983333333334,
    #  'Verbs': 0.9995933333333333,
    #  'We': 0.9853266666666667,
    #  'Work': 0.9972916666666667,
    #  'You': 0.8662666666666666}

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
        X_train_unigrams.append(articles_training[article_id].clean_article())
        # Extract title features
        X_train_titles.append(articles_training[article_id].title)
        # Extract structural features
        X_train_structure.append([articles_training[article_id].count_quotes,
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

        X_train_liwc.append(liwc)
        X_train_liwc_title.append(liwc_title)

        # Extract punctuation features
        punc = []
        punc_title = []
        punc_order = []
        for feature in dataprocessor.punctuations.keys():
            tf_article = 0
            tf_title = 0
            idf_title = 0
            idf_article = 0
            if articles_training[article_id].all_punc != 0:
                tf_article = articles_training[article_id].punctuation_counts[feature] / articles_training[
                    article_id].all_punc
            if articles_training[article_id].all_punc_title != 0:
                tf_title = articles_training[article_id].punctuation_counts_title[feature] / articles_training[
                    article_id].all_punc_title
            punc.append(tf_article * idf_punc_article[feature])
            punc_title.append(tf_title * idf_punc_title[feature])
            punc_order.append(feature)
        X_train_punctuation.append(punc)
        X_train_punctuation_title.append(punc_title)

        y_train.append(articles_training[article_id].hyperpartisan)

    save_dataset([id_train, X_train_unigrams, y_train],
                 '/tmp/hyperpartisan_project/project/new_data/pkl-objects/train_unigrams.pkl')
    save_dataset([id_train, X_train_structure, y_train],
                 '/tmp/hyperpartisan_project/project/new_data/pkl-objects/train_structure.pkl')
    save_dataset([id_train, X_train_liwc, y_train],
                 '/tmp/hyperpartisan_project/project/new_data/pkl-objects/train_liwc.pkl')
    save_dataset([id_train, X_train_punctuation, y_train],
                 '/tmp/hyperpartisan_project/project/new_data/pkl-objects/train_punctuation.pkl')

    save_dataset([id_train, X_train_titles, y_train],
                 '/tmp/hyperpartisan_project/project/new_data/pkl-objects/train_titles.pkl')
    save_dataset([id_train, X_train_liwc_title, y_train],
                 '/tmp/hyperpartisan_project/project/new_data/pkl-objects/train_liwc_title.pkl')
    save_dataset([id_train, X_train_punctuation_title, y_train],
                 '/tmp/hyperpartisan_project/project/new_data/pkl-objects/train_punctuation_title.pkl')

    save_dataset(
        [[key for key in dataprocessor.liwc_features.keys()], [key for key in dataprocessor.punctuations.keys()]],
        "/tmp/hyperpartisan_project/project/new_data/pkl-objects/features_order.pkl")

    # Test Data extraction:
    articles_testing = dataprocessor.construct_article_array(
        "/tmp/hyperpartisan_project/project/new_data/articles-validation-bypublisher-20181122.xml",
        "/tmp/hyperpartisan_project/project/new_data/ground-truth-validation-bypublisher-20181122.xml")

    X_test_unigrams = []
    X_test_titles = []
    X_test_liwc = []
    X_test_punctuation = []
    X_test_liwc_title = []
    X_test_punctuation_title = []
    X_test_structure = []
    id_test = []
    y_test = []

    for article_id in articles_testing:
        id_test.append(article_id)
        # Extract unigram features
        X_test_unigrams.append(articles_testing[article_id].clean_article())
        # Extract title features
        X_test_titles.append(articles_testing[article_id].title)
        # Extract structural features
        X_test_structure.append([articles_testing[article_id].count_quotes,
                                 articles_testing[article_id].count_paragraphs,
                                 articles_testing[article_id].count_urls])
        # Extract liwc features
        liwc = []
        liwc_title = []
        liwc_order = []
        for feature in dataprocessor.liwc_features.keys():
            tf_article = 0
            tf_title = 0
            if articles_testing[article_id].all_liwc != 0:
                tf_article = articles_testing[article_id].liwc_counts[feature] / articles_testing[article_id].all_liwc
            if articles_testing[article_id].all_liwc_title != 0:
                tf_title = articles_testing[article_id].liwc_counts_title[feature] / articles_testing[
                    article_id].all_liwc_title

            liwc.append(tf_article * idf_liwc_article[feature])
            liwc_title.append(tf_title * idf_liwc_title[feature])
            liwc_order.append(feature)

        X_test_liwc.append(liwc)
        X_test_liwc_title.append(liwc_title)

        # Extract punctuation features
        punc = []
        punc_title = []
        punc_order = []
        for feature in dataprocessor.punctuations.keys():
            tf_article = 0
            tf_title = 0
            idf_title = 0
            idf_article = 0
            if articles_testing[article_id].all_punc != 0:
                tf_article = articles_testing[article_id].punctuation_counts[feature] / articles_testing[
                    article_id].all_punc
            if articles_testing[article_id].all_punc_title != 0:
                tf_title = articles_testing[article_id].punctuation_counts_title[feature] / articles_testing[
                    article_id].all_punc_title
            punc.append(tf_article * idf_punc_article[feature])
            punc_title.append(tf_title * idf_punc_title[feature])
            punc_order.append(feature)
        X_test_punctuation.append(punc)
        X_test_punctuation_title.append(punc_title)

        y_test.append(articles_testing[article_id].hyperpartisan)

    save_dataset([id_test, X_test_unigrams, y_test],
                 '/tmp/hyperpartisan_project/project/new_data/pkl-objects/test_unigrams.pkl')
    save_dataset([id_test, X_test_structure, y_test],
                 '/tmp/hyperpartisan_project/project/new_data/pkl-objects/test_structure.pkl')
    save_dataset([id_test, X_test_liwc, y_test],
                 '/tmp/hyperpartisan_project/project/new_data/pkl-objects/test_liwc.pkl')
    save_dataset([id_test, X_test_punctuation, y_test],
                 '/tmp/hyperpartisan_project/project/new_data/pkl-objects/test_punctuation.pkl')

    save_dataset([id_test, X_test_titles, y_test],
                 '/tmp/hyperpartisan_project/project/new_data/pkl-objects/test_titles.pkl')
    save_dataset([id_test, X_test_liwc_title, y_test],
                 '/tmp/hyperpartisan_project/project/new_data/pkl-objects/test_liwc_title.pkl')
    save_dataset([id_test, X_test_punctuation_title, y_test],
                 '/tmp/hyperpartisan_project/project/new_data/pkl-objects/test_punctuation_title.pkl')
