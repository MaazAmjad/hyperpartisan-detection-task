# add unigrams+bigrams
# + new dictionary lueke
# + wordnet effect (64+).
#
# save it as sparse matrix and then pickle.
# + SVM with linear kernel.
# use lueke for sintement analysis.
# 1. convert everything to unigrams(TF-IDF encoding)
# 2. for every group of features "class" have a model (LR and SVM)
# 3. have 1 model that combines all features (LR and SVM)
from pickle import dump
from xml.dom import minidom

from sklearn.feature_extraction.text import TfidfVectorizer

from articleclass import ArticleClass


# save a dataset to file
def save_dataset(dataset, filename):
    dump(dataset, open(filename, 'wb'))
    print('Saved: %s' % filename)


class DataProcessing(object):
    def __init__(self):
        self.vectorizer = None
        pass

    def read_articles(self, articles_filename, labels_filename, training=True):
        # parse an xml file by name
        if training:
            vocabulary = []
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
            if training:
                vocabulary.extend(articles[article_id].text)
        for label in articles_labels:
            article_id = label.attributes['id'].value
            articles[article_id].hyperpartisan = (1 if label.attributes['hyperpartisan'].value == "true" else 0)
            articles[article_id].bias = label.attributes['bias'].value
            articles[article_id].labeled_by = label.attributes['labeled-by'].value
        print("Done reading data")
        if training:
            self.vectorizer = self.extract_features(vocabulary)
            print("Done extracting features")

        return articles

    def extract_features(self, vocabulary):
        vectorizer = TfidfVectorizer()
        vectorizer.fit(vocabulary)
        return vectorizer


if __name__ == '__main__':
    dataprocessor = DataProcessing()
    articles_training = dataprocessor.read_articles("/tmp/pycharm_project_127/data/articles-training-20180831.xml",
                                                    "/tmp/pycharm_project_127/data/ground-truth-training-20180831.xml",
                                                    training=True)
    articles_testing = dataprocessor.read_articles("/tmp/pycharm_project_127/data/articles-validation-20180831.xml",
                                                   "/tmp/pycharm_project_127/data/ground-truth-validation-20180831.xml",
                                                   training=False)
    # articles = dataprocessor.read_articles("/tmp/pycharm_project_127/data/test/articles-training-text.xml",
    #                                       "/tmp/pycharm_project_127/data/test/articles-training.xml", training=True)
    X_train = []
    y_train = []
    X_test = []
    y_test = []

    # for id in articles:
    #    X_train.append(" ".join(articles[id].text))
    #    y_train.append(articles[id].hyperpartisan)

    for id in articles_training:
        X_train.append(" ".join(articles_training[id].text))
        y_train.append(articles_training[id].hyperpartisan)

    for id in articles_testing:
        X_test.append(" ".join(articles_testing[id].text))
        y_test.append(articles_testing[id].hyperpartisan)

    print("Done appending labels")

    X_train = dataprocessor.vectorizer.transform(X_train)
    X_test = dataprocessor.vectorizer.transform(X_test)

    save_dataset([X_train, y_train], 'pkl-objects/train.pkl')
    save_dataset([X_test, y_test], 'pkl-objects/test.pkl')
    save_dataset(dataprocessor.vectorizer, 'pkl-objects/vectorizer.pkl')

    print("Done vectorizing")
