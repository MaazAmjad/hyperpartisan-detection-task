from string import punctuation
from typing import List, Any

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import sent_tokenize


class ArticleClass:
    def __init__(self, liwc_dict, punc_dict):
        self.id = None
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
        self.liwc_counts = dict.fromkeys(liwc_dict, 0)
        self.punctuation_counts = dict.fromkeys(punc_dict, 0)
        self.liwc_counts_title = dict.fromkeys(liwc_dict, 0)
        self.punctuation_counts_title = dict.fromkeys(punc_dict, 0)
        self.all_liwc = 0
        self.all_punc = 0
        self.all_liwc_title = 0
        self.all_punc_title = 0

    # turn a doc into clean tokens (code from:
    # https://machinelearningmastery.com/develop-word-embedding-model-predicting-movie-review-sentiment/) 
    def clean_article(self):
        """
        This method cleans the article and stores its text as an array of tokens. 
        """
        # split into tokens by white space
        tokens = self.text.split(" ")
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
        return tokens

    def split_to_sentences(self):
        sent_tokenize_list = sent_tokenize(self.text)
        self.text = sent_tokenize_list

    def clean_title(self):
        """
        This method cleans the article and stores its text as an array of tokens.
        """
        # split into tokens by white space
        tokens = self.title.split(" ")
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
        self.title = tokens
