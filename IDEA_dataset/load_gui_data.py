import time
load_start_time = time.time()

import csv
import nltk
import re
import numpy as np


import gensim, logging


from gensim.models import Word2Vec

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer



###load data
with open('./android/ebay/total_info.txt') as f:
    reader = csv.reader(f, delimiter="\t")
    d = list(reader)


reviews = []


for line in d:
    vals = line[0].split("******")
    reviews.append(vals[1])

    
    
stop_words = stopwords.words('english')
def custom_preprocessor(text):
    porter = PorterStemmer()

    #split into sentences
    sentences = sent_tokenize(text)
    
    final_sentences = []
    
    for sentence in sentences:
        sentence_split = sentence.split(" ")
        
        #remove words in not in stop words, and make lowercase
        words = [word.lower() for word in sentence_split if word.lower() not in stop_words]
        #get rid of words with non alphanumeric characters in it
        #(should we replace these with a token?)
        words = [word for word in words if word.isalpha()]
        #stem words
        words = [porter.stem(word) for word in words]

        final_sentences.append(" ".join(words))
        
        #consider joining sentences with a stop token
    return " ".join(final_sentences), final_sentences, sentences


processed_reviews = []
for review in reviews:
    processed_review = custom_preprocessor(review) 
    processed_reviews.append(processed_review[0])

    
    
#get rid of reviews that are empty after preprocessing
#(not that many)

processed_review_lens = np.array([len(review) for review in [r.split(" ") for r in processed_reviews]])
#if using stop tokens "<END>" then empty reviews have a length of 6
nonzero_indeces = np.where(processed_review_lens > 0)


final_reviews =  [review.split(" ") for review in np.array(processed_reviews)[nonzero_indeces]]




start = time.time()
model = Word2Vec(final_reviews, min_count=1)
model.save("../../large files/ebay.model")
print(time.time() - start)