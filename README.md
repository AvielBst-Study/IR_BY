# IR_BY

Build new index for body/title/anchor using the tokenizer we asked you to use
Focus on one search function at a time
Use classes that we used in previous HW: 
	read_posting_list
	Reader/Writer
Consider writing to more than 124 buckets to reduce overhead while searching
	
# 28/12
Used assignment3 code to read wikidumps from Nir's bucket into local folder "/wikidumps_test"
used spark.read to turn each file into rdd
extracted each term's posting list by using `def read_posting_list(inverted, w)`