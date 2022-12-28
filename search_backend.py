import pandas as pd
import numpy as np
import json
from nltk.corpus import stopwords


class BackEnd:
    def __init__(self):
        self.terms_in_train_set = []
        self.stopwords = frozenset(stopwords.words('english'))

    def get_train_query_terms(self):
        """
        Using queries_train.json file
        Returns:
            list of unique terms (not stopwords) that appear in one of the train queries
        """
        with open("queries_train.json") as f:
            json_data = json.load(f)
            terms_in_train_set = []
            for query, wiki_id in json_data.items():
                terms = query.split()
                terms_in_train_set += [t if t[-1] != '?' else t[:-1] for t in terms]
            self.terms_in_train_set = list(set([term.lower() for term in terms_in_train_set if term not in self.stopwords]))
        return self.terms_in_train_set


def main():
    operator = BackEnd()
    train_terms = operator.get_train_query_terms()
    print(train_terms)


if __name__ == '__main__':
    main()
