import numpy as np
from gensim import models
from scipy.sparse import hstack
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC

from preprocessing.data_processing import load_dataset, count_features, save_dataset


def classifier_init(name):
    if name == "linear_regression":
        return LogisticRegression()
    elif name == "SVM":
        return LinearSVC()
    else:
        return None


def load_data(feature_name, train, test):
    vocabulary = load_dataset("pkl-objects/" + feature_name + ".pkl")
    vectorizer = count_features(vocabulary)
    save_dataset(vectorizer, 'pkl-objects/' + feature_name + '_vectorizer.pkl')
    train = vectorizer.transform(train)
    test = vectorizer.transform(test)
    return train, test


def classify(feature_name, classifier_name, train, test, train_label, test_label, cfname="results_classification.csv"):
    classifier = classifier_init(classifier_name)
    classifier.fit(train, train_label)

    y_pred = classifier.predict(test)
    int_predict = y_pred.ravel()
    np.savetxt(cfname, np.dstack((np.arange(1, int_predict.size + 1), int_predict, test_label))[0], "%d,%d,%d",
               header="index,prediction,actual")

    classification_result = open("figs/classification_report_" + classifier_name + "_" + feature_name + ".tsv", "w")
    classification_result.write(classification_report(test_label, y_pred))
    np.savetxt("coeff_" + cfname, classifier.coef_, delimiter=',')  # X is an array

    print("Wait here")



def feature_classification(feature):
    X_train, y_train = load_dataset("pkl-objects/train_v.pkl")
    X_test, y_test = load_dataset("pkl-objects/test_v.pkl")
    X_train = [' '.join(row) for row in X_train]
    X_test = [' '.join(row) for row in X_test]
    X_train_v, X_test_v = load_data(feature, X_train, X_test)
    save_dataset([X_train_v, y_train], 'pkl-objects/train_' + feature + '.pkl')
    save_dataset([X_test_v, y_test], 'pkl-objects/test_' + feature + '.pkl')
    classifier_n = "linear_regression"
    classify(feature, classifier_n, X_train_v, X_test_v, y_train, y_test)
    classifier_n = "SVM"
    classify(feature, classifier_n, X_train_v, X_test_v, y_train, y_test)


def combined_classification():
    # X_train_tfidf, y_train = load_dataset("pkl-objects/train_tfidf.pkl")
    # X_test_tfidf, y_test = load_dataset("pkl-objects/test_tfidf.pkl")
    X_train_liwc, y_train = load_dataset("pkl-objects/train_liwc.pkl")
    X_test_liwc, y_test = load_dataset("pkl-objects/test_liwc.pkl")
    # X_train_hedges, y_train = load_dataset("pkl-objects/train_titles.pkl")
    # X_test_hedges, y_test = load_dataset("pkl-objects/test_titles.pkl")
    X_train_vad, y_train = load_dataset("pkl-objects/X_train_topics.pkl")
    X_test_vad, y_test = load_dataset("pkl-objects/X_test_topics.pkl")

    X_train = hstack([X_train_liwc, X_train_vad])
    X_test = hstack([X_test_liwc, X_test_vad])
    feature = "combined_nt_with_topics"
    classifier_n = "linear_regression"
    classify(feature, classifier_n, X_train, X_test, y_train, y_test, "results30_classification_combined.csv")
    # classifier_n = "SVM"
    # classify(feature, classifier_n, X_train, X_test, y_train, y_test)


def classification_with_topic_liwc():
    X_train_liwc, y_train = load_dataset("pkl-objects/train_liwc.pkl")
    X_test_liwc, y_test = load_dataset("pkl-objects/test_liwc.pkl")
    X_train_topics = load_dataset("corpus.pkl")
    X_train_topics = features_padding(doc_topics=X_train_topics, n_topics=30)
    save_dataset([X_train_topics, y_train], "pkl-objects/X_train_topics.pkl")
    X_test_topics, y_test = load_dataset("pkl-objects/X_test_topics.pkl")
    X_train = hstack([X_train_liwc, X_train_topics])
    X_test = hstack([X_test_liwc, X_test_topics])
    feature = "combined_lwic_topics30_friday"
    classifier_n = "linear_regression"
    classify(feature, classifier_n, X_train, X_test, y_train, y_test, "results30_classification_liwc.csv")
    # classifier_n = "SVM"
    # classify(feature, classifier_n, X_train, X_test, y_train, y_test)


def classification_with_topic():
    X_train_liwc, y_train = load_dataset("pkl-objects/train_liwc.pkl")
    X_test_liwc, y_test = load_dataset("pkl-objects/test_liwc.pkl")
    X_train_topics = load_dataset("corpus.pkl")
    X_train_topics = features_padding(doc_topics=X_train_topics, n_topics=30)
    X_test_topics = load_dataset("test_corpus.pkl")
    X_test_topics = features_padding(doc_topics=X_test_topics, n_topics=30)
    X_train = X_train_topics
    X_test = X_test_topics
    feature = "topics30_friday"
    classifier_n = "linear_regression"
    classify(feature, classifier_n, X_train, X_test, y_train, y_test, "results30_classification.csv")
    # classifier_n = "SVM"
    # classify(feature, classifier_n, X_train, X_test, y_train, y_test)


def features_padding(doc_topics, n_topics):
    topic_distribution = np.zeros(shape=(len(doc_topics), n_topics))
    i = 0
    print(len(doc_topics))

    lda = models.LdaModel.load("../model/lda30.model")
    lda_doc_topics = lda[doc_topics]
    for row in lda_doc_topics:
        for (topic_num, prop_topic) in row:
            topic_distribution[i][topic_num] = prop_topic
        i += 1
    return topic_distribution


if __name__ == '__main__':
    combined_classification()
    # feature_classification("vad")
    # classification_with_topic_liwc()
    # classification_with_topic()
