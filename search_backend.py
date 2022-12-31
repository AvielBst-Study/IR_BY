import json
from nltk.corpus import stopwords
import pickle
import gcsfs
from collections import defaultdict
from inverted_index_gcp import *


class BackEnd:
    def __init__(self):
        self.terms_in_train_set = []
        self.stopwords = frozenset(stopwords.words('english'))
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
        # Create a GCSFS filesystem object
        fs = gcsfs.GCSFileSystem()

        files = fs.ls(f"gs://{self.bucket_name}/postings_gcp/")
        files = [f for f in files if f.endswith('.pickle')]
        super_posting_locs = defaultdict(list)
        for file in files:
            with fs.open(f"gs://{file}", "rb") as f:
                # Load the data from the pickle file into a dictionary
                posting_locs_list = pickle.load(f)

            # merge the posting locations into a single dict and run more tests (5 points)
            for k, v in posting_locs_list.items():
                # super_posting_locs[k].extend(v)
                if k in super_posting_locs:
                    super_posting_locs[k] = super_posting_locs.get(k) + v
                else:
                    super_posting_locs[k] = v

        return super_posting_locs

    def read_postings(self):
        pass

def main():
    operator = BackEnd()
    train_terms = operator.get_posting_locs_from_pkls()
    # print(train_terms)


if __name__ == '__main__':
    main()
