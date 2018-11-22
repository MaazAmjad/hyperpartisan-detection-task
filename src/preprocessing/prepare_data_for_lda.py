import scipy
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

from preprocessing.data_processing import load_dataset


def save_vocab_to_text(filename, data):
    with open(filename, 'w') as text_file:
        text_file.write(data)


def transform_articles_to_counts(vocab, articles, include_ones):
    count_vectorizer = CountVectorizer()
    count_vectorizer.fit(vocab)
    transformed_articles = []
    # [(Article index, Word index, Word count)]
    for i in range(len(articles)):
        article = articles[i]
        counts = scipy.sparse.coo_matrix(count_vectorizer.transform([" ".join(article)]))
        for k, j, v in zip(counts.row, counts.col, counts.data):
            if not include_ones:
                if v > 1:
                    transformed_articles.append("{} {} {}".format(i + 1, j + 1, v))
            else:
                transformed_articles.append("{} {} {}".format(i + 1, j + 1, v))

    return transformed_articles


def save_docwords(filename, data, d, w, nnz):
    with open(filename, 'w') as text_file:
        text_file.write("{}\n".format(d))
        text_file.write("{}\n".format(w))
        text_file.write("{}\n".format(nnz))
        text_file.write(data)


if __name__ == '__main__':
    # 1. Get the vectorizer and extract its features = vocabulary.
    # 2. Write vocabulary to file : vocab.articles.txt
    # 3. Get Each article and extract its words, map it to the vocabulary features and count them.
    # 4. Write documents words to file: docword.articles.txt

    # 1
    vectorizer = load_dataset('pkl-objects/unigram_vectorizer.pkl')
    vocabulary = vectorizer.get_feature_names()

    # 2
    stop_words = set(stopwords.words('english'))
    filtered_vocab = [w for w in vocabulary if not w in stop_words]
    save_vocab_to_text('vocab.articles.txt', "\n".join(filtered_vocab))

    # 3
    training_articles = load_dataset('pkl-objects/train_v.pkl')[0]
    # [(Article index, Word index, Word count)]
    docwords = transform_articles_to_counts(filtered_vocab, training_articles, True)
    docwords_more_than_one = transform_articles_to_counts(filtered_vocab, training_articles, False)
    save_docwords('docword.articles.txt', "\n".join(docwords),
                  len(training_articles),
                  len(filtered_vocab),
                  len(docwords))

    save_docwords('docword_more_than_one.articles.txt', "\n".join(docwords_more_than_one),
                  len(training_articles),
                  len(filtered_vocab),
                  len(docwords_more_than_one))
