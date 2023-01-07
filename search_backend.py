from inverted_index_gcp import *
import datetime
import json
from nltk.corpus import stopwords
import pickle
import gcsfs
from collections import defaultdict
import numpy as np
import math
from scipy.sparse import csr_matrix, lil_matrix
import scipy.sparse
import scipy.linalg
from nltk.stem.porter import *


class Data:
    def __init__(self):
        self.terms_in_train_set = []
        self.stopwords = frozenset(stopwords.words('english'))
        self.GCSFS = gcsfs.GCSFileSystem()
        self.inverted = InvertedIndex()
        self.bucket_name = "206224503_ir_hw3"
        # with open("pr_dict.pkl", "rb") as f:
        #     self.pr_dict = pickle.load(f)
        with self.GCSFS.open(r"gs://ir_project_utils_files/DL_dict.pkl", "rb") as f:
            self.DL = pickle.load(f)
        with self.GCSFS.open(r"gs://ir_project_utils_files/doc_title_dict.pickle", "rb") as f:
            self.doc_title_dict = pickle.load(f)
        with self.GCSFS.open(r"gs://ir_project_utils_files/docs_norm_dict.pkl", "rb") as f:
            self.doc_norm_dict = pickle.load(f)

        english_stopwords = frozenset(stopwords.words('english'))
        corpus_stopwords = ["category", "references", "also", "external", "links",
                            "may", "first", "see", "history", "people", "one", "two",
                            "part", "thumb", "including", "second", "following",
                            "many", "however", "would", "became"]

        self.all_stopwords = english_stopwords.union(corpus_stopwords)
        self.RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
        self.term_dict = {}


class BackEnd:
    def __init__(self, path, Data, part: str):  # part = body/title/anchor
        self.part = part
        self.Data = Data
        self.read_index(path)


    def get_pr(self, wiki_ids):
        wiki_ids_str = list(map(str, wiki_ids))
        return [(wid, self.Data.pr_dict[wid]) for wid in wiki_ids_str if wid in self.Data.pr_dict]

    def read_index(self, index_file: pickle):
        with open(index_file, 'rb') as f:
            data = pickle.load(f)
        self.Data.inverted.df = data.df
        self.Data.inverted.term_total = data.posting_locs
        self.Data.inverted.posting_locs = data.posting_locs
        self.Data.term_dict = {term: idx for idx, term in
         enumerate(self.Data.inverted.term_total)}

    def tokenize(self, text, use_stemming=False):
        tokens = [token.group() for token in self.Data.RE_WORD.finditer(text.lower())]
        tokens = [token for token in tokens if (token not in self.Data.all_stopwords)]
        if use_stemming:
            tokens = [self.Data.stemmer.stem(token) for token in tokens]
        return tokens

    def get_top_n(self, sim_list, n):
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
        if n > 0:
            return sim_list[:n]
        else:
            return sim_list

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
                tf = counter[token]  # term frequency divded by the length of the query
                df = index.df[token]
                idf = math.log((len(self.Data.DL)) / (df + epsilon), 10)  # smoothing
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
        3 arrays of candidates. In the following format: term_array, document_ids, document_scores

        """
        term_array = None
        document_ids = None
        document_scores = None
        for term in np.unique(query_to_search):
            if term in words:

                list_of_doc = pls[words.index(term)]
                normalized_tfidf = [(doc_id, freq * math.log(len(self.Data.DL) / index.df[term], 10))
                                    for doc_id, freq in list_of_doc]

                # create the doc_id array and score array
                cur_document_ids, cur_document_scores = zip(*normalized_tfidf)
                cur_document_ids, cur_document_scores = np.asarray(cur_document_ids), np.asarray(cur_document_scores)

                # create term_id array for each term, that contains term_id in size of the number of docs it had
                term_id = self.Data.term_dict[term]
                cur_term_array = np.full(len(cur_document_ids), fill_value= term_id )

                # create or concatenate each of the three
                if term_array is None:
                    term_array = cur_term_array
                    document_scores = cur_document_scores
                    document_ids = cur_document_ids
                else:
                    term_array = np.concatenate([term_array,cur_term_array])
                    document_scores = np.concatenate([document_scores, cur_document_scores])
                    document_ids = np.concatenate([document_ids, cur_document_ids])

        return term_array, document_ids, document_scores


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

        term_id, candidate_id, candidate_score = self.get_candidate_documents_and_scores(query_to_search, index, words, pls)

        total_vocab_size = len(index.term_total)
        max_doc_id = np.max(candidate_id)#TODO change to doc_id dictionary
        if max_doc_id is None:
            return []
        D = scipy.sparse.coo_matrix((candidate_score, (candidate_id, term_id)),
                                    shape = (max_doc_id + 1, total_vocab_size + 1))
        return D



    def score(self,D,Q):
        """gets a doc tfidf matrix and retruns the values of the similarity score of D and Q
        -------------
        """
        dot = D._mul_vector(Q)
        query_norm = scipy.linalg.norm(Q)
        scores = [(doc_id, dot[doc_id]/(query_norm*doc_norm))for doc_id, doc_norm in self.Data.doc_norm_dict.items() if doc_id < dot.shape[0] and dot[doc_id]/(query_norm*doc_norm) > 0]

        return scores

    def get_topN_score_for_queries(self, queries_to_search, index, n):
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
        for query_id, tokens in queries_to_search.items():
            t1 = datetime.datetime.now()
            words, pls = self.Data.inverted.get_posting_iter(index, tokens, self.part)
            print(f"after change : {datetime.datetime.now() - t1}")

            if len(pls[0]) == 0:
                return []
            D = self.generate_document_tfidf_matrix(tokens, index, words, pls)
            vect_query = self.generate_query_tfidf_vector(tokens, index)
            scores = self.score(D,vect_query)
            sorted_result = sorted(scores, key=lambda x: x[1], reverse=True)
            retrieved_docs = [(str(doc_id), str(score), self.Data.doc_title_dict[doc_id]) for doc_id, score in sorted_result]
            return self.get_top_n(retrieved_docs, n)

    def get_titles(self, query_to_search, index):
        """
        ---------
        returns: the titles by the order of distinct term frequency in the query
        """
        words, pls = self.Data.inverted.get_posting_iter(index, query_to_search, self.part)
        term_array, document_ids, document_scores = self.get_candidate_documents_and_scores(query_to_search, index, words, pls)
        title_rank_dict = Counter(document_ids)
        sorted_results = [(str(doc_id), str(score),self.Data.doc_title_dict[doc_id]) for doc_id , score in title_rank_dict.most_common()]

        return sorted_results

    def activate_search(self, query, n=0):
        tokenized_query = self.tokenize(query)
        return self.get_topN_score_for_queries({0: tokenized_query}, self.Data.inverted, n)

    def activate_title_search(self,query):
        tokenized_query = self.tokenize(query)
        return self.get_titles(tokenized_query,self.Data.inverted)


def main():
    data_obj = Data()
    operator = BackEnd(r"title_index.pkl", data_obj, "title")

    query = "Ciggarets"
    # query = ""
    t1 = datetime.datetime.now()
    result = operator.activate_title_search(query)
    # result = operator.activate_search(query)
    if len(result) == 0:
        print('got no res')
        return 0
    t2 = datetime.datetime.now() - t1
    print(f"run in {t2}")



if __name__ == '__main__':
    main()


#######drppped###########
    # with open("queries_train.json", 'r') as f:
    #     queries = json.load(f)
    # acc_scores = []
    # TOTAL_TIME = datetime.timedelta()
    # # for query, real in queries.items():
    # # print(f"Search for query: {query}")
    # query = "Ciggarets"
    # t1 = datetime.datetime.now()
    # result = operator.activate_search(query, 20)
    # if len(result) == 0:
    #     print('got no res')
    #     return 0
    # t2 = datetime.datetime.now() - t1
    # TOTAL_TIME += t2
    # # result = set([int(element[0]) for element in result])
    # # inters = result.intersection(set(real))
    # # acc_scores.append(len(inters)/len(result))
    # # print(f"    Time:{t2}\n    Precision:{len(inters)/len(result)}\n    Right Docs:{inters}")
    # print(f"\n{'*'*20}\nMAP@10: {np.mean(acc_scores)}\nTOTAL TIME: {TOTAL_TIME}")