class Article:
    def __init__(self, id):
        self.id = id
        self.hyperpartisan = "hyperpartisan"
        self.bias = "bias"
        self.title = "title"
        self.text = []
        self.labeled_by = "labeled_by"
        self.published_at = "published_at"
        self.count_urls = 0
        self.count_paragraphs = 0
        self.count_quotes = 0
        self.hedges = []
        self.boosters = []
        self.negatives = []
        self.positives = []
