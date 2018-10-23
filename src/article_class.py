from string import punctuation

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from typing import List, Any


class ArticleClass:
    def __init__(self, id):
        self.id = id
        self.hyperpartisan = "hyperpartisan"
        self.bias = "bias"
        self.title = "title"
        self.text = ""
        self.labeled_by = "labeled_by"
        self.published_at = "published_at"
        self.count_urls = 0
        self.count_paragraphs = 0
        self.count_quotes = 0
        self.hedges = []
        self.boosters = []
        self.negatives = []
        self.positives = []

    # turn a doc into clean tokens (code from: 
    # https://machinelearningmastery.com/develop-word-embedding-model-predicting-movie-review-sentiment/) 
    def clean_article(self):
        """
        This method cleans the article and stores its text as an array of tokens. 
        """
        # split into tokens by white space
        tokens = self.text.split()
        # remove punctuation from each token
        table = str.maketrans('', '', punctuation)
        tokens = [w.translate(table) for w in tokens]  # type: List[Any]
        # remove remaining tokens that are not alphabetic
        tokens = [word for word in tokens if word.isalpha()]
        # filter out stop words
        stop_words = set(stopwords.words('english'))
        tokens = [w for w in tokens if not w in stop_words]
        # lemmatization and lowercase
        lmtzr = WordNetLemmatizer()
        tokens = [lmtzr.lemmatize(w.lower()) for w in tokens]
        # filter out short tokens
        tokens = [word for word in tokens if len(word) > 1]
        self.text = tokens
