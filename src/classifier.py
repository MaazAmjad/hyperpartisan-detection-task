from scipy.sparse import hstack
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC

from data_processing import load_dataset, extract_features, save_dataset


def classifier_init(name):
    if name == "linear_regression":
        return LogisticRegression()
    elif name == "SVM":
        return LinearSVC()
    else:
        return None


def load_data(feature_name, train, test):
    vocabulary = load_dataset("pkl-objects/" + feature_name + ".pkl")
    vectorizer = extract_features(vocabulary)
    save_dataset(vectorizer, 'pkl-objects/' + feature_name + '_vectorizer.pkl')
    train = vectorizer.transform(train)
    test = vectorizer.transform(test)
    return train, test


def classify(feature_name, classifier_name, train, test, train_label, test_label):
    classifier = classifier_init(classifier_name)
    classifier.fit(train, train_label)
    y_pred = classifier.predict(test)
    classification_result = open("figs/classification_report_" + classifier_name + "_" + feature_name + ".tsv", "w")
    classification_result.write(classification_report(test_label, y_pred))


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
    X_train_hedges, y_train = load_dataset("pkl-objects/train_hedges.pkl")
    X_test_hedges, y_test = load_dataset("pkl-objects/test_hedges.pkl")
    X_train_opinion, y_train = load_dataset("pkl-objects/train_opinion.pkl")
    X_test_opinion, y_test = load_dataset("pkl-objects/test_opinion.pkl")
    X_train_vad, y_train = load_dataset("pkl-objects/train_vad.pkl")
    X_test_vad, y_test = load_dataset("pkl-objects/test_vad.pkl")

    X_train = hstack([X_train_liwc, X_train_hedges, X_train_opinion, X_train_vad])
    X_test = hstack([X_test_liwc, X_test_hedges, X_test_opinion, X_test_vad])
    feature = "combined_nt"
    classifier_n = "linear_regression"
    classify(feature, classifier_n, X_train, X_test, y_train, y_test)
    classifier_n = "SVM"
    classify(feature, classifier_n, X_train, X_test, y_train, y_test)


if __name__ == '__main__':
    combined_classification()
    # feature_classification("vad")