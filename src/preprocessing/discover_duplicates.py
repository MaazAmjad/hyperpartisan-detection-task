from preprocessing.data_processing import load_dataset

if __name__ == '__main__':
    classification_type = ["article", "publisher"]
    # Load All datasets:
    id_train_by_article, X_train_titles_by_article, y_train_by_article = load_dataset(
        '../new_data/pkl-objects/train_titles_' + classification_type[0] + '.pkl')
    id_train_by_publisher, X_train_titles_by_publisher, y_train_by_publisher = load_dataset(
        '../new_data/pkl-objects/train_titles_' + classification_type[1] + '.pkl')
    id_test_by_publisher, X_test_titles_by_publisher, y_test_by_publisher = load_dataset(
        '../new_data/pkl-objects/test_titles_' + classification_type[1] + '.pkl')

    duplicate_articles_indexes = []
    for title in X_train_titles_by_article:
        if title in X_train_titles_by_publisher:
            index_article = X_train_titles_by_article.index(title)
            index_pub = X_train_titles_by_publisher.index(title)
            publisher_tuple = (id_train_by_publisher[index_pub], y_train_by_publisher[index_pub])
            article_tuple = (id_train_by_article[index_article], y_train_by_article[index_article])
            duplicate_articles_indexes.append((publisher_tuple, article_tuple))
        elif title in X_test_titles_by_publisher:
            index_article = X_train_titles_by_article.index(title)
            index_pub = X_test_titles_by_publisher.index(title)
            publisher_tuple = (id_test_by_publisher[index_pub], y_test_by_publisher[index_pub])
            article_tuple = (id_train_by_article[index_article], y_train_by_article[index_article])
            duplicate_articles_indexes.append((publisher_tuple, article_tuple))
