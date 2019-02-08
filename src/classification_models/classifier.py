from joblib import dump
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from preprocessing.data_processing import load_dataset


def classifier_init(name):
    """
    Initiate the chosen classifier
    :param name: classifier name which can be either,
    * svm: support vector machine
    * gb: gradient boosting
    * tree: decision tree
    :return: the chosen classifier or none, if classifier name is wrong
    """
    seed = 7
    if name == "svm":
        return SVC(kernel="linear", random_state=seed, probability=True)
    elif name == "gb":
        num_trees = 200
        model = GradientBoostingClassifier(n_estimators=num_trees, random_state=seed)
        return model
    elif name == "tree":
        cart = DecisionTreeClassifier()
        num_trees = 200
        model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
        return model
    else:
        return None


def classify(classifier_name, train, train_label):
    """
    This function initiates the classifier and fit it to training data
    :param classifier_name: classifier name which can be either,
    * svm: support vector machine
    * gb: gradient boosting
    * tree: decision tree
    :param train: training data
    :param train_label: truth value of training data
    :return: trained classifier
    """
    model = classifier_init(classifier_name)
    model.fit(train, train_label)
    return model


def vote_classify(train, train_label):
    """
    This method uses voting classifier which combines 3 classifiers and returns their best vote
    :param train:
    :param train_label:
    :return: trained vote classifier
    """
    # create the sub models
    estimators = []
    model1 = classifier_init("svm")
    estimators.append(('svm', model1))
    model2 = classifier_init("gb")
    estimators.append(('gb', model2))
    model3 = classifier_init("tree")
    estimators.append(('tree', model3))
    # create the ensemble model
    ensemble = VotingClassifier(estimators=estimators)
    ensemble_p = Pipeline(
        [('scalar', RobustScaler(with_centering=False)), ('chi2', SelectKBest(chi2, k=300)), ('clf', ensemble)])
    ensemble_p.fit(train, train_label)
    return ensemble_p


if __name__ == '__main__':
    classification_type = "article"
    # Load All datasets:
    id_train, x_train_unigrams, y_train = load_dataset(
        '../new_data/pkl-objects/train_unigrams_' + classification_type + '.pkl')
    _, x_train_structure, _ = load_dataset(
        '../new_data/pkl-objects/train_structure_' + classification_type + '.pkl')
    _, x_train_liwc, _ = load_dataset(
        '../new_data/pkl-objects/train_liwc_' + classification_type + '.pkl')
    _, x_train_punctuation, _ = load_dataset(
        '../new_data/pkl-objects/train_punctuation_' + classification_type + '.pkl')
    # _, X_train_titles, _ = load_dataset(
    #    '../new_data/pkl-objects/train_titles_' + classification_type + '.pkl')
    # _, X_train_liwc_title, _ = load_dataset(
    #    '../new_data/pkl-objects/train_liwc_title_' + classification_type + '.pkl')
    # _, X_train_punctuation_title, _ = load_dataset(
    #    '../new_data/pkl-objects/train_punctuation_title_'
    #    + classification_type + '.pkl')
    liwc_feature_names, punc_feature_names = load_dataset(
        "../new_data/pkl-objects/features_order_" + classification_type + ".pkl")
    # articles_emotions = load_dataset("../new_data/pkl-objects/article_emotions_byarticle.pkl")
    # titles_emotions = load_dataset("../new_data/pkl-objects/titles_emotions_byarticle.pkl")

    # Create TFIDF unigrams
    articles_vectorizer = TfidfVectorizer(stop_words="english")
    x_train_tfidf_articles = articles_vectorizer.fit_transform([" ".join(row) for row in x_train_unigrams])
    # titles_vectorizer = TfidfVectorizer(stop_words="english")
    # X_train_tfidf_titles = titles_vectorizer.fit_transform(X_train_titles)

    # Start code for classification:
    x_train_combined = hstack([x_train_tfidf_articles, x_train_liwc, x_train_punctuation, x_train_structure])
    SVM = Pipeline([('svd', TruncatedSVD(n_components=1000)),
                    ('clf', classifier_init("svm"))]).fit(x_train_combined, y_train)

    # Save the classifiers to use later in classification tasks
    dump(SVM, "../output/by"
         + classification_type + "_SVM_classification_model.pkl")
    dump(articles_vectorizer,
         "../output/by"
         + classification_type + "_tfidf_articles_model.pkl")
    # dump(titles_vectorizer,
    #     "../output/by"
    #     + classification_type + "_tfidf_titles_model.pkl")
