from flask import Flask, request, jsonify
from search_backend import *


class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


with open("IR/DL_dict.pkl", "rb") as f:
    DL = pickle.load(f)
with open("IR/pr_dict.pkl", "rb") as f:
    pr_dict = pickle.load(f)
with open("IR/doc_title_dict.pickle", "rb") as f:
    doc_title_dict = pickle.load(f)
with open("IR/docs_norm_dict.pkl", "rb") as f:
    doc_norm_dict = pickle.load(f)
with open("IR/pageviews-202108-user.pkl", "rb") as f:
    pv_dict = pickle.load(f)

app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
data_body = Data()
data_title = Data()
data_anchor = Data()
body_operator = BackEnd(data_body, "body", pr_dict, pv_dict, DL, doc_title_dict, doc_norm_dict)
title_operator = BackEnd(data_title, "title", pr_dict, pv_dict, DL, doc_title_dict, doc_norm_dict)
anchor_operator = BackEnd(data_anchor, "anchor", pr_dict, pv_dict, DL, doc_title_dict, doc_norm_dict)


@app.route("/search")
def search():
    """ Returns up to a 100 of your best search results for the query. This is
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    """
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    weights = [1.2, 130, 130]
    res_body = body_operator.activate_search(query)
    res_title = title_operator.activate_title_search(query)
    res_anchor = anchor_operator.activate_title_search(query)

    # add the scores
    docs = {}
    # body scores
    for doc in res_body:
        docs[doc[0]] = weights[0] * float(doc[1])
    # title scores
    for doc in res_title:
        if doc[0] in docs:
            docs[doc[0]] += weights[1] * float(doc[1])
        else:
            docs[doc[0]] = weights[1] * float(doc[1])

    # anchor scores
    for doc in res_anchor:
        if doc[0] in docs:
            docs[doc[0]] += weights[2] * float(doc[1])
        else:
            docs[doc[0]] = weights[2] * float(doc[1])

    docs = [(doc_id, score) for doc_id, score in docs.items()]
    docs = sorted(docs, key=lambda x: x[1], reverse=True)

    # change get document title from id

    res = docs[:40]
    res = [(doc_id, title_operator.doc_title_dict[int(doc_id)]) for doc_id, _ in res]

    if len(res) < 5:
        return res_body[:20]
    # END SOLUTION
    return jsonify(res)


@app.route("/search_body")
def search_body():
    """ Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the
        staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    """
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    res = body_operator.activate_search(query, n=10)
    res = [(doc_id, title_operator.doc_title_dict[int(doc_id)]) for doc_id, _ in res]
    # END SOLUTION
    return jsonify(res)


@app.route("/search_title")
def search_title():
    """ Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF
        DISTINCT QUERY WORDS that appear in the title. DO NOT use stemming. DO
        USE the staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords. For example, a document
        with a title that matches two distinct query words will be ranked before a
        document with a title that matches only one distinct query word,
        regardless of the number of times the term appeared in the title (or
        query).

        Test this by navigating to the URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    """
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    res = title_operator.activate_title_search(query)
    res = [(doc_id, title_operator.doc_title_dict[int(doc_id)]) for doc_id, _ in res]
    # END SOLUTION
    return jsonify(res)


@app.route("/search_anchor")
def search_anchor():
    """ Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE ANCHOR TEXT of articles, ordered in descending order of the
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page.
        DO NOT use stemming. DO USE the staff-provided tokenizer from Assignment
        3 (GCP part) to do the tokenization and remove stopwords. For example,
        a document with an anchor text that matches two distinct query words will
        be ranked before a document with anchor text that matches only one
        distinct query word, regardless of the number of times the term appeared
        in the anchor text (or query).

        Test this by navigating to the URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    """
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    res = anchor_operator.activate_title_search(query)
    res = [(doc_id, title_operator.doc_title_dict[int(doc_id)]) for doc_id, _ in res]
    # END SOLUTION
    return jsonify(res)


@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    """ Returns PageRank values for a list of provided wiki article IDs.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correspond to the provided article IDs.
    """
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    wiki_ids = list(map(str, wiki_ids))
    res = [(int(wid), pr_dict[wid]) for wid in wiki_ids if wid in pr_dict]
    # END SOLUTION
    return jsonify(res)


@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    """ Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correspond to the
          provided list article IDs.

          [13, 2222, 15]
          [pageview(13), pageview(2222), pageview(15)]
    """
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION

    for doc_id in wiki_ids:
        res.append(pv_dict[int(doc_id)])

    # END SOLUTION
    return jsonify(res)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)
