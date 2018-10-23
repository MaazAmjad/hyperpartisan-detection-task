from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC

from data_processing import load_dataset
from dictionary_features import features_list


def classifier(name):
    if name == "linear_regression":
        return LogisticRegression()
    elif name == "SVM":
        return LinearSVC()
    else:
        return None


if __name__ == '__main__':
    classifier_name = "linear_regression"
    classifier = classifier(classifier_name)
    for key in features_list:
        X_train, y_train = load_dataset("pkl-objects/train_" + key + ".pkl")
        X_test, y_test = load_dataset("pkl-objects/test_" + key + ".pkl")
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        # conf_matrix = confusion_matrix(y_test, y_pred)
        classification_result = open("figs/classification_report_" + classifier_name + "_" + key + ".tsv", "w")
        classification_result.write(classification_report(y_test, y_pred))
