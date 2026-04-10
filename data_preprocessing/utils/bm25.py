import bm25s
import Stemmer


class BM25Retriever:
    """BM25 retriever backed by bm25s with Snowball stemming for Russian."""

    def __init__(self):
        self.retriever = None
        self.stemmer = Stemmer.Stemmer('russian')

    def index(self, corpus_dicts: list[dict]) -> None:
        """Build a BM25 index from a list of document dicts.

        Args:
            corpus_dicts: Each dict must have at least a ``text`` field.
        """
        # Tokenize on the text field; the full dicts are stored as the corpus
        # so that retrieve() returns original objects rather than raw strings.
        corpus_texts = [doc['text'] for doc in corpus_dicts]
        corpus_tokens = bm25s.tokenize(corpus_texts, stopwords="ru", stemmer=self.stemmer)

        self.retriever = bm25s.BM25(corpus=corpus_dicts)
        self.retriever.index(corpus_tokens)

    def search(self, queries: list[str], k: int = 30) -> list:
        """Retrieve the top-k documents for each query.

        Args:
            queries: List of query strings.
            k:       Maximum number of results per query.

        Returns:
            2-D array of matching document dicts, shape [len(queries), k].
        """
        query_tokens = bm25s.tokenize(queries, stopwords="ru", stemmer=self.stemmer)
        docs, scores = self.retriever.retrieve(query_tokens, k=k)
        return docs