
from scipy.sparse import csr_matrix

from preprocessing.data_processing import save_dataset, load_dataset
from preprocessing.dictionary_features import liwc_features_list, NRC_VAD_Lexicon_features_list


def count_features(filename, feature_vectorizer):
    features_file = open("/tmp/pycharm_project_127/data/dictionary/" + filename + ".csv",
                         encoding="latin-1").readlines()
    transformed_features = feature_vectorizer.transform(features_file)
    return csr_matrix(transformed_features.sum(axis=0)).sign()


def map_features_to_articles(articles, features_matrix):
    new_articles = articles.multiply(features_matrix)
    return new_articles


def read_features(filename):
    features_file = open("/tmp/pycharm_project_127/data/dictionary/" + filename + ".csv",
                         encoding="latin-1").readlines()
    return features_file


def extract_features_from_list(features_list, name):
    features_vocabulary = []
    for key, value in features_list.items():
        features_vocabulary.extend(read_features(value))
    save_dataset(features_vocabulary, 'pkl-objects/' + name + '.pkl')


def old_map_features_to_liwc():
    vectorizer = load_dataset("pkl-objects/vectorizer.pkl")
    X_train, y_train = load_dataset("pkl-objects/train_v.pkl")
    X_test, y_test = load_dataset("pkl-objects/test_v.pkl")
    for key, value in liwc_features_list.items():
        try:
            features = load_dataset('pkl-objects/' + key + '.pkl')
        except OSError:
            features = count_features(value, vectorizer)
            save_dataset(features, 'pkl-objects/' + key + '.pkl')
        X_train_new = map_features_to_articles(X_train, features)
        X_test_new = map_features_to_articles(X_test, features)
        save_dataset([X_train_new, y_train], 'pkl-objects/train_' + key + '.pkl')
        save_dataset([X_test_new, y_test], 'pkl-objects/test_' + key + '.pkl')


if __name__ == '__main__':
    # extract_features_from_list(liwc_features_list, "liwc")
    # extract_features_from_list(hedges_features_list, "hedges")
    # extract_features_from_list(opinion_lexicon_features_list, "opinion")
    extract_features_from_list(NRC_VAD_Lexicon_features_list, "vad")
