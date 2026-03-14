import bm25s
import Stemmer


class BM25Retriever:
    def __init__(self):
        self.retriever = None
        self.stemmer = Stemmer.Stemmer('russian')

    def index(self, corpus_dicts):
        # Extract text field for tokenization
        corpus_texts = [doc['text'] for doc in corpus_dicts]
        corpus_tokens = bm25s.tokenize(corpus_texts, stopwords="ru", stemmer=self.stemmer)

        # Pass the dictionaries themselves as the corpus
        self.retriever = bm25s.BM25(corpus=corpus_dicts)
        self.retriever.index(corpus_tokens)

    def search(self, queries, k=30):
        query_tokens = bm25s.tokenize(queries, stopwords="ru", stemmer=self.stemmer)
        docs, scores = self.retriever.retrieve(query_tokens, k=k)
        return docs