# from inverted_index_gcp import *
from inverted_index_colab import *
import json
from nltk.corpus import stopwords
import pickle
import gcsfs
from collections import defaultdict


class BackEnd:
    def __init__(self):
        self.terms_in_train_set = []
        self.stopwords = frozenset(stopwords.words('english'))
        self.GCSFS = gcsfs.GCSFileSystem()
        self.inverted = InvertedIndex()
        self.bucket_name = "206224503_ir_hw3"

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
            self.terms_in_train_set = list(set([t.lower() for t in terms_in_train_set if t not in self.stopwords]))
        return self.terms_in_train_set

    def get_posting_locs_from_pkls(self):
        files = self.GCSFS.ls(f"gs://{self.bucket_name}/postings_gcp/")
        files = [f for f in files if f.endswith('.pickle')]
        super_posting_locs = defaultdict(list)
        for file in files:
            with self.GCSFS.open(f"gs://{file}", "rb") as f:
                # Load the data from the pickle file into a dictionary
                posting_locs_list = pickle.load(f)

            for k, v in posting_locs_list.items():
                if k in super_posting_locs:
                    super_posting_locs[k] = super_posting_locs.get(k) + v
                else:
                    super_posting_locs[k] = v

        return super_posting_locs

    @staticmethod
    def read_index(index_file: pickle):
        with open(index_file, 'rb') as f:
            data = pickle.load(f)
        return data

    @staticmethod
    def read_posting_list(inverted, w):
        with closing(MultiFileReader()) as reader:
            locs = inverted.posting_locs[w]
            b = reader.read(locs, inverted.df[w] * TUPLE_SIZE)
            posting_list = []
            for i in range(inverted.df[w]):
                doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
                tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
                posting_list.append((doc_id, tf))
            return posting_list


def main():
    operator = BackEnd()
    # operator.inverted.posting_locs = operator.get_posting_locs_from_pkls()
    inverted = operator.read_index(r"index.pkl")

    train_terms = operator.get_train_query_terms()
    for term in train_terms:
        print(f"{term}\n  Posting list: {operator.read_posting_list(inverted, term)}")


if __name__ == '__main__':
    main()
