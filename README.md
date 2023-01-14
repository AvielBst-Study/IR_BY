# BY - Search Engine
by: Aviel Ben Siman Tov and Ken Yaggel
## search_frontend
On the top of the script, we load all relevant files we prepared beforehand
* DL - document's lengths
* pr_dict - page rank of each documents 
* pv_dict - page views of each document
* doc_title_dict - title of each document
* doc_norm_dict - norma of each document's tf-idf vector

### search()
Returns all relevant documents from each partial search engine in this script (title/body/anchor)
Then using our weights to calculate new scores for each of them
Returning the top 40

## search_backend()
To organize our code, we created 2 classes:
	- Data: utility class holding index and global statistics for each partial
	- BackEnd: operates the search. contains most of the functions the engine uses.

In BackEnd:
### get_topN_scores_for_query()
Input:
	- query
	- n: determines the number of documents to retrieve
Output:
	list of tuples (doc_id, sim_score) sorted by descending order by scores.

### get_candidates_documents_and_scores()
Input:
	- query
	- index
	- words: terms in the query
	- pls: posting lists of each of the terms in the query
Output:
	Returns 3 arrays in size of total scores. 
	Array for terms, documents and for scores.
	Afterwards these arrays are being used to create the coo_matrix
	
### generate_document_tfidf_matrix()
Input:
	- query
	- index
	- words
	- pls
Output:
	Generates a coo_matrix of tf-idf scores in size of max(candidate_id)
	
### generate_query_tfidf_vector
Function taken from homework
Creates an tf-idf vector of the query

### score()
Calculates Cosine Similarity

### get_titles()
Returns a sorted list of tuples (doc_id, score) where number of query terms in the document determines the score
This function being used for title_search/anchor_search


	


