from inverted_index_gcp import *
import datetime
import random

# from inverted_index_colab import *
import json
from nltk.corpus import stopwords
import pickle
import gcsfs
from collections import defaultdict
import numpy as np
import math
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


from scipy.sparse import csr_matrix,lil_matrix,coo_matrix
# TODO NEED TO FIX INDEX.PKL - IT HAS NO TOTAL_TERM FIELD
# TODO ADD TITLES TO POSTING LISTS SO IT WILL BE EASY TO PULL WHEN NEW QUERY ARRIVES
# TODO change D in generate_document_tfidf_matrix to be initiated in the __init__
# TODO change
# TODO set dict generate_document_tfidf_matrix to of vectorizer 

class BackEnd:
    def __init__(self, path):
        self.terms_in_train_set = []
        self.stopwords = frozenset(stopwords.words('english'))
        self.GCSFS = gcsfs.GCSFileSystem()
        self.inverted = InvertedIndex()
        self.bucket_name = "206224503_ir_hw3"
        self.read_index(path)
        with self.GCSFS.open(r"gs://ir_training_index/doc_dl_dict.pickle", "rb") as f:
            self.DL = pickle.load(f)
        with self.GCSFS.open(r"gs://ir_training_index/doc_title_dict.pickle", "rb") as f:
            self.doc_title_dict = pickle.load(f)
        self.max_doc = max(list(self.DL.keys()))


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
            self.terms_in_train_set = list(
                set([t.lower() for t in terms_in_train_set if t.lower() not in self.stopwords]))

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

    def read_index(self, index_file: pickle):
        with open(index_file, 'rb') as f:
            data = pickle.load(f)
        self.inverted.df = data.df
        self.inverted.term_total = data.posting_locs
        self.inverted.posting_locs = data.posting_locs

    @staticmethod
    def read_posting_list(inverted, w):
        with closing(MultiFileReader()) as reader:
            locs = inverted.posting_locs[w]
            try:
                b = reader.read(locs, inverted.df[w] * TUPLE_SIZE)
            except KeyError:
                return []

            posting_list = []
            for i in range(inverted.df[w]):
                doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
                tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
                posting_list.append((doc_id, tf))
            return posting_list

    def get_top_n(self, sim_dict, N=3):
        """
        Sort and return the highest N documents according to the cosine similarity score.
        Generate a dictionary of cosine similarity scores

        Parameters:
        -----------
        sim_dict: a dictionary of similarity score as follows:
                                                                    key: document id (e.g., doc_id)
                                                                    value: similarity score. We keep up to 5 digits after the decimal point. (e.g., round(score,5))

        N: Integer (how many documents to retrieve). By default N = 3

        Returns:
        -----------
        a ranked list of pairs (doc_id, score) in the length of N.
        """
        x = 10
        return sorted([(doc_id, score) for doc_id, score in sim_dict.items()], key=lambda x: x[1],
                      reverse=True)[:N]

    def generate_query_tfidf_vector(self, query_to_search, index):
        """
        Generate a vector representing the query. Each entry within this vector represents a tfidf score.
        The terms representing the query will be the unique terms in the index.

        We will use tfidf on the query as well.
        For calculation of IDF, use log with base 10.
        tf will be normalized based on the length of the query.

        Parameters:
        -----------
        query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                        Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

        index:           inverted index loaded from the corresponding files.

        Returns:
        -----------
        vectorized query with tfidf scores
        """
        epsilon = .0000001
        total_vocab_size = len(index.term_total)
        Q = np.zeros(total_vocab_size)
        term_vector = list(index.term_total.keys())
        counter = Counter(query_to_search)
        for token in np.unique(query_to_search):
            if token in index.term_total.keys():  # avoid terms that do not appear in the index.
                tf = counter[token] / len(query_to_search)  # term frequency divded by the length of the query
                df = index.df[token]
                idf = math.log((len(self.DL)) / (df + epsilon), 10)  # smoothing


                ind = term_vector.index(token)
                Q[ind] = tf * idf

        return Q

    def get_candidate_documents_and_scores(self, query_to_search, index, words, pls):
        """
        Generate a dictionary representing a pool of candidate documents for a given query. This function will go through every token in query_to_search
        and fetch the corresponding information (e.g., term frequency, document frequency, etc.') needed to calculate TF-IDF from the posting list.
        Then it will populate the dictionary 'candidates.'
        For calculation of IDF, use log with base 10.
        tf will be normalized based on the length of the document.

        Parameters:
        -----------
        query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                        Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

        index:           inverted index loaded from the corresponding files.

        words,pls: iterator for working with posting.

        Returns:
        -----------
        dictionary of candidates. In the following format:
                                                                key: pair (doc_id,term)
                                                                value: tfidf score.
        """
        candidates = {}
        for term in np.unique(query_to_search):
            if term in words:
                list_of_doc = pls[words.index(term)]
                normlized_tfidf = [(doc_id, (freq / self.DL[doc_id]) * math.log(len(self.DL) / index.df[term], 10))
                                   for doc_id, freq in list_of_doc]

                for doc_id, tfidf in normlized_tfidf:
                    candidates[(doc_id, term)] = candidates.get((doc_id, term), 0) + tfidf
        return candidates

    def generate_document_tfidf_matrix(self, query_to_search, index, words, pls):
        """
        Generate a DataFrame `D` of tfidf scores for a given query.
        Rows will be the documents candidates for a given query
        Columns will be the unique terms in the index.
        The value for a given document and term will be its tfidf score.

        Parameters:
        -----------
        query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                        Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

        index:           inverted index loaded from the corresponding files.


        words,pls: iterator for working with posting.

        Returns:
        -----------
        DataFrame of tfidf scores.
        """

        
        print('got here')

        total_vocab_size = len(index.term_total)

        candidates_scores = self.get_candidate_documents_and_scores(query_to_search, index, words, pls)
        # unique_candidates = np.unique(candidates_scores.keys(),lambda x:x[0])
        unique_candidates = np.unique([doc_id for doc_id, freq in candidates_scores.keys()])
        t = datetime.datetime.now()
        D = lil_matrix((len(candidates_scores), total_vocab_size))
        print(f'creating the lil took:{datetime.datetime.now()-t}')
        # candidates_id = list()

        cloumns_dict = {term: idx for idx, term in enumerate(index.term_total)}
        candidates_dict = {key : candidate_id for candidate_id, key in enumerate(unique_candidates) } #maps from doc_id to candidate_id

        for idx, key in enumerate(candidates_scores):
            tfidf = candidates_scores[key]
            doc_id, term = candidates_dict[key[0]], cloumns_dict[key[1]]

            # candidates_id.append(doc_id)
            D[doc_id, term] = tfidf

        return csr_matrix(D), candidates_dict#candidates_id

    # def cosine_similarity(self, D, Q):
    #     """
    #     Calculate the cosine similarity for each candidate document in D and a given query (e.g., Q).
    #     Generate a dictionary of cosine similarity scores
    #     key: doc_id
    #     value: cosine similarity score
    #
    #     Parameters:
    #     -----------
    #     D: DataFrame of tfidf scores.
    #
    #     Q: vectorized query with tfidf scores
    #
    #     Returns:
    #     -----------
    #     dictionary of cosine similarity score as follows:
    #                                                                 key: document id (e.g., doc_id)
    #                                                                 value: cosine similarty score.
    #     """
    #     # YOUR CODE HERE
    #     scores = {}
    #     for doc_id, tfidf_vect in D.iterrows():
    #         doc_tfidf = np.array(tfidf_vect)
    #         numerator = np.sum(doc_tfidf * Q)
    #         denominator = np.linalg.norm(doc_tfidf) * np.linalg.norm(Q)
    #         scores[doc_id] = numerator / denominator
    #     return scores

    def get_topN_score_for_queries(self, queries_to_search, index, query, N=3):
        """
            Generate a dictionary that gathers for every query its topN score.

            Parameters:
            -----------
            queries_to_search: a dictionary of queries as follows:
                                                                key: query_id
                                                                value: list of tokens.
            index:           inverted index loaded from the corresponding files.
            N: Integer. How many documents to retrieve. This argument is passed to the topN function. By default N = 3, for the topN function.

            Returns:
            -----------
            return: a dictionary of queries and topN pairs as follows:
                                                                key: query_id
                                                                value: list of pairs in the following format:(doc_id, score).
        """
        # Get iterator to work with posting lists
        print('getting posting iter')
        t1 = datetime.datetime.now()
        words, pls = self.inverted.get_posting_iter(index, query)
        print(f"get_posting_iter took {datetime.datetime.now()-t1}")
        retrieved_docs = {}
        for query_id, tokens in queries_to_search.items():
            t1 = datetime.datetime.now()
            D, candidate_list = self.generate_document_tfidf_matrix(tokens, index, words, pls)
            print(f"generate_document_tfidf_matrix: {datetime.datetime.now() - t1}")
            vect_query = self.generate_query_tfidf_vector(tokens, index).reshape(1, -1)


            retrieved_docs[query_id] = self.get_top_n(dict(list(zip(candidate_list, D._mul_vector(vect_query)))),N)
            # retrieved_docs[query_id] = self.get_top_n(dict(list(zip(candidate_list, cosine_similarity(D, vect_query)))), N)

        return retrieved_docs

    def activate_search(self, query, N=50):
        doc_score_dic = self.get_topN_score_for_queries({0: query}, self.inverted, query, N=N)
        # return doc_score_dic
        return [(id, score, self.doc_title_dict[id]) for id, score in doc_score_dic[0]]


def main():#
    operator = BackEnd(r"index.pkl")
    # Generate a random query
    t1 = datetime.datetime.now()
    possible_query_terms = operator.get_train_query_terms()
    query = random.sample(possible_query_terms, 3)
    query = ["best", "marvel", "movie"]
    result = operator.activate_search(query)
    print(f"\n\nQuery: {query}\nTook {datetime.datetime.now() - t1}\nRelevant Docs are:")
    for id, score, title in result:
        print(f"    Id:{id}, Score:{score}, title:{title}")


if __name__ == '__main__':
    main()
