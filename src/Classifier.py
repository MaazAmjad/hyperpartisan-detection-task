from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC

from data_processing import load_dataset


def classifier(name):
    if name == "linear-regression":
        return LogisticRegression()
    elif name == "SVM":
        return LinearSVC()
    else:
        return None


if __name__ == '__main__':
    X_train, y_train = load_dataset("train.pkl")
    X_test, y_test = load_dataset("test.pkl")
    classifier = classifier("linear-regression")
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    confusion_matrix = confusion_matrix(y_test, y_pred)
    print(classification_report(y_test, y_pred))
