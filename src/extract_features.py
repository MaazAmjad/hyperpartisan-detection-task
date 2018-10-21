import itertools

from scipy.sparse import csr_matrix

from data_processing import save_dataset, load_dataset


def count_features(filename, feature_vectorizer):
    features_file = open("/tmp/pycharm_project_127/data/dictionary/" + filename + ".csv").readlines()
    transformed_features = feature_vectorizer.transform(features_file)
    return transformed_features


def map_features_to_articles(articles, labels, features_matrix):
    articles_iterator = articles.tocoo()
    features_iterator = features_matrix.tocoo()
    row_ind = []
    col_ind = []
    data = []
    y = []
    for k, l, m in itertools.izip(articles_iterator.row, articles_iterator.col, articles_iterator.data):
        for i, j, v in itertools.izip(features_iterator.row, features_iterator.col, features_iterator.data):
            if l == j:
                row_ind = k
                col_ind = l
                data = m
                y.append(labels[k])

    articles = csr_matrix((data, (row_ind, col_ind)), shape=(len(row_ind), len(col_ind)))
    return articles, y


if __name__ == '__main__':
    vectorizer = load_dataset("pkl-objects/vectorizer.pkl")
    features_list = dict(Funct="1", Pronoun="2", Ppron="3", I="4", We="5", You="6", SheHe="7", They="8", Ipron="9",
                         Article="10", Verbs="11", AuxVb="12", Past="13", Present="14", Future="15", Adverbs="16",
                         Prep="17", Conj="18", Negate="19", Quant="20", Numbers="21", Swear="22", Social="23",
                         Family="24", Friends="25", Humans="26", Affect="27", Posemo="28", Negemo="29", Anx="30",
                         Anger="30", Sad="31", CogMech="32", Insight="33", Cause="34", Discrep="35", Tentat="36",
                         Certain="37", Inhib="38", Incl="39", Excl="40", Percept="41", See="42", Hear="43", Feel="44",
                         Bio="45", Body="46", Health="47", Sexual="48", Ingest="49", Relativ="50", Motion="51",
                         Space="52", Time="53", Work="54", Achiev="55", Leisure="56", Home="57", Money="58", Relig="59",
                         Death="60", Assent="61", Nonflu="62", Filler="63")

    features = count_features(features_list["Funct"], vectorizer)
    X_train, y_train = load_dataset("pkl-objects/train.pkl")
    X_test, y_test = load_dataset("pkl-objects/test.pkl")

    X_train_new, y_train_new = map_features_to_articles(X_train, y_train, features)
    X_test_new, y_test_new = map_features_to_articles(X_test, y_test, features)

    save_dataset([X_train_new, y_train_new], 'pkl-objects/train_funct.pkl')
    save_dataset([X_test_new, y_test_new], 'pkl-objects/test_funct.pkl')
    save_dataset(features, 'pkl-objects/funct.pkl')
