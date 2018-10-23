
from scipy.sparse import csr_matrix

from data_processing import save_dataset, load_dataset
from dictionary_features import features_list


def count_features(filename, feature_vectorizer):
    features_file = open("/tmp/pycharm_project_127/data/dictionary/" + filename + ".csv",
                         encoding="latin-1").readlines()
    transformed_features = feature_vectorizer.transform(features_file)
    return csr_matrix(transformed_features.sum(axis=0)).sign()


def map_features_to_articles(articles, features_matrix):
    new_articles = articles.multiply(features_matrix)
    return new_articles


if __name__ == '__main__':
    vectorizer = load_dataset("pkl-objects/vectorizer.pkl")
    X_train, y_train = load_dataset("pkl-objects/train.pkl")
    X_test, y_test = load_dataset("pkl-objects/test.pkl")
    for key, value in features_list.items():
        try:
            features = load_dataset('pkl-objects/' + key + '.pkl')
        except OSError:
            features = count_features(value, vectorizer)
            save_dataset(features, 'pkl-objects/' + key + '.pkl')
        X_train_new = map_features_to_articles(X_train, features)
        X_test_new = map_features_to_articles(X_test, features)
        save_dataset([X_train_new, y_train], 'pkl-objects/train_' + key + '.pkl')
        save_dataset([X_test_new, y_test], 'pkl-objects/test_' + key + '.pkl')
