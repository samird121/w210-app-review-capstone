import time
load_start_time = time.time()
import csv
import nltk
import re
import numpy as np
import pandas as pd
from dateutil import parser

import gensim, logging
from gensim.models import Word2Vec

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.preprocessing import StandardScaler

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



filepaths = ['./android/ebay/total_info.txt', './android/clean_master/total_info.txt', './android/swiftkey_keyboard/total_info.txt', './android/viber/total_info.txt', './ios/noaa-radar-pro-severe-weather/total_info.txt', './ios/youtube/total_info.txt']
app_names = ['ebay', 'clean_master', 'swiftkey_keyboard', 'viber', 'noaa-radar-pro-severe-weather', 'youtube']
oses = ['android', 'android', 'android', 'android', 'ios', 'ios']


###load data
for i in range(len(filepaths)): 

    filepath = filepaths[i]
    app_name = app_names[i]
    os = oses[i]
    print('Loading data for', app_name, '........')
    
    with open(filepath) as f:
        reader = csv.reader(f, delimiter="\t")
        d = list(reader)


    ratings = []
    reviews = []
    titles = []
    dates = []
    versions = []
    
    #account for the different file structures for android/ios apps
    if os == 'android':
        for line in d:
            vals = line[0].split("******")
            ratings.append(float(vals[0]))
            reviews.append(vals[1])
            dates.append(vals[2])
            versions.append(vals[3])
    elif os == 'ios':
        for line in d:
            vals = line[0].split("******")
            ratings.append(float(vals[0]))
            reviews.append(vals[1])
            #ios review dates are like 'Apr 01, 2017'
            #we'll turn them into '2017-04-01' to make it consistent with the android ones
            date = parser.parse(vals[3]).strftime('%Y-%m-%d')
            dates.append(date)
            versions.append(vals[4])
        








    processed_reviews = []
    for review in reviews:
        processed_review = custom_preprocessor(review) 
        processed_reviews.append(processed_review[0])



    #get rid of reviews that are empty after preprocessing
    #(not that many)

    processed_review_lens = np.array([len(review) for review in [r.split(" ") for r in processed_reviews]])
    #if using stop tokens "<END>" then empty reviews have a length of 6
    nonzero_indeces = np.where(processed_review_lens > 1)
    #print(len(nonzero_indeces))
    #print(min(processed_review_lens))

    final_reviews_processed = np.array(processed_reviews)[nonzero_indeces]
    final_reviews =  [review.split(" ") for review in final_reviews_processed]
    final_reviews_unprocessed =  np.array(reviews)[nonzero_indeces]
    final_ratings = [float(rating) for rating in np.array(ratings)[nonzero_indeces]]
    #final_titles = np.array(titles)[nonzero_indeces]
    final_dates = np.array(dates)[nonzero_indeces]
    unique_dates = np.unique(np.array(final_dates))
    unique_date_indices = []
    for date in unique_dates:
            date_indices = np.where(np.array(final_dates)==date)[0]
            unique_date_indices.append(date_indices)
    final_versions = np.array(versions)[nonzero_indeces]

    model = Word2Vec(final_reviews, min_count=1)
    model_filename = "../../large files/"+app_name+".model"
    model.save(model_filename)


    d = {'reviews_unprocessed':final_reviews_unprocessed, 'reviews_processed':final_reviews_processed, 'dates':final_dates, 'versions':final_versions, 'ratings':final_ratings}

    df = pd.DataFrame(data = d)
    data_filename = "../../large files/"+app_name+"_post_processing_data.csv"
    df.to_csv(data_filename)



print('All done!!')