from pickle import load

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# load a clean dataset
def load_dataset(filename):
    return load(open(filename, 'rb'))


if __name__ == '__main__':
    X_train, y_train = load_dataset("train.pkl")
    X_test, y_test = load_dataset("test.pkl")
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    confusion_matrix = confusion_matrix(y_test, y_pred)
    print(classification_report(y_test, y_pred))
