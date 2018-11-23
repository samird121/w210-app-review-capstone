print('Loading GUI...')

import time
load_start_time = time.time()

import csv
import nltk
import re
import numpy as np
import datetime

import gensim, logging
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
#from sklearn.cluster import KMeans
#from sklearn.decomposition import PCA

from scipy.spatial.distance import cdist, cosine
#from scipy.spatial import cKDTree

from gensim.models import Word2Vec

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from bokeh.io import curdoc
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import Slider, TextInput, PreText, Select, RangeSlider, Button
from bokeh.plotting import figure





###load data
with open('./android/ebay/total_info.txt') as f:
    reader = csv.reader(f, delimiter="\t")
    d = list(reader)

ratings = []
reviews = []
titles = []
dates = []
versions = []

for line in d:
    vals = line[0].split("******")
    ratings.append(float(vals[0]))
    reviews.append(vals[1])
    dates.append(vals[2])
    versions.append(vals[3])
    
    
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
processed_sentences = []
raw_sentences = []
for review in reviews:
    processed_review = custom_preprocessor(review) 
    processed_reviews.append(processed_review[0])
    processed_sentences.append(processed_review[1])
    raw_sentences.append(processed_review[2])
    
    
#get rid of reviews that are empty after preprocessing
#(not that many)

processed_review_lens = np.array([len(review) for review in [r.split(" ") for r in processed_reviews]])
#if using stop tokens "<END>" then empty reviews have a length of 6
nonzero_indeces = np.where(processed_review_lens > 0)


final_reviews =  [review.split(" ") for review in np.array(processed_reviews)[nonzero_indeces]]
final_reviews_unprocessed =  np.array(reviews)[nonzero_indeces]
final_ratings = np.array(ratings)[nonzero_indeces]
#final_titles = np.array(titles)[nonzero_indeces]
final_dates = np.array(dates)[nonzero_indeces]
unique_dates = np.unique(np.array(final_dates))
unique_date_indices = []
for date in unique_dates:
        date_indices = np.where(np.array(final_dates)==date)[0]
        unique_date_indices.append(date_indices)
final_versions = np.array(versions)[nonzero_indeces]




model_ebay = Word2Vec.load("../../large files/ebay.model")
model_youtube = Word2Vec.load("../../large files/youtube.model")



#here we create a vector for each review,
#which will be the simple average of all word vectors in that review.
#these vectors will then be used for clustering, data reduction, etc.
avg_vectors = []
for review in final_reviews:
    avg_vectors.append(np.mean([model.wv[word] for word in review], axis=0))
    
avg_vectors = np.array(avg_vectors)  


final_review_lengths = [len(review.split(" ")) for review in final_reviews_unprocessed]

#scaling 
scaler = StandardScaler()
avg_vectors_scaled = scaler.fit_transform(avg_vectors)
print(avg_vectors.shape)
print(avg_vectors_scaled.shape)

def find_relevant_reviews(key, avg_vectors, raw_text, scaled=False):
    
    indices = None
    distances = None
    
    key_processed = custom_preprocessor(key)[0]
    key_list = key_processed.split(' ')
    #filter to only those in the w2v model's covacbulary
    vocab = model.wv.vocab.keys()
    key_list_vocab_words = []
    key_list_nonvocab_words = []
    for word in key_list:
        if word in vocab:
            key_list_vocab_words.append(word)
        else:
            key_list_nonvocab_words.append(word)
    
    #only move on if the list isn't empty (i.e. if the key had no words in the vocab)
    if len(key_list_vocab_words) > 0:
        key_vector = np.mean([model.wv[word] for word in key_list_vocab_words if word in model.wv.vocab.keys()], axis=0)
        
        if scaled:
            key_vector = scaler.transform(np.array(key_vector).reshape(1,-1))

        distances = [1-cosine(key_vector, vector) for vector in avg_vectors]
        indices = np.argsort(distances)
    else:
        print("Warning: none of the words in the query were in the model's vocabulary.")
    
    if len(key_list_nonvocab_words) > 0:
        print("Excluded words:", key_list_nonvocab_words)
        
    return indices, distances
    
def print_relevant_reviews(indices, distances, reviews, n=10):
    text = "Most relevant reviews in selected timeframe:\n\n"
    top_indices = indices[-n:]
    for i in range(len(top_indices)):
        index = top_indices[-i-1]
        distance = distances[index]
        text += "'"+str(reviews[index])+", Cosine similarity: "+str(round(distance, 4))+")\n"
        #text += "'"+str(final_reviews_unprocessed[index])+"' (Rating: "+str(final_ratings[index])+")\n"
        
    return text
    

    
def get_topic_evolution_data(distances, query, relevance_threshold = 0.8):
    
    
    mean_distances = []
    n_relevant_reviews = []
    percent_relevant_reviews= []
    mean_ratings = []
    
    for date_indices in unique_date_indices:
        date_distances = np.array(distances)[date_indices]
        mean_distances.append(np.mean(date_distances))
        
        relevant_indeces = np.where(date_distances > relevance_threshold)[0]
        n = len(relevant_indeces)
        percent = 100*(n/len(date_indices))

        n_relevant_reviews.append(n)
        percent_relevant_reviews.append(percent)
        
        date_ratings = np.array(final_ratings)[date_indices][relevant_indeces]
        mean_ratings.append(np.mean(date_ratings))
       
    return mean_distances, percent_relevant_reviews

    

    
    
    
    
#initialize variables for gui
#interesting ones: 'fast forward button', 'crash', 'freeze', 'update'
query = 'i love this app'

indices, distances = find_relevant_reviews(query, avg_vectors_scaled, final_reviews_unprocessed,  scaled=True)
mean_distances, percent_relevant_reviews = get_topic_evolution_data(distances, query)


#indices, distances = find_relevant_reviews(query, avg_vectors, final_reviews_unprocessed, n=10, scaled=False)
#for i in range(len(indices)):
#    index = indices[-(i+1)]
#    print(final_reviews_unprocessed[index])
#    print("Rating:", final_ratings[index])
#    print("Cosine similarity:", round(distances[index], 4))
#    print('*******\n')
    
    
    
    
    
    
    
    
    
#BOKEH!
#combining automatic and user-defined queries:
#find most important topic with automated topics, then show graphs of those
#how would a company use this
    
    
    
    
    
# Set up data

source = ColumnDataSource(data=dict(x=range(len(unique_dates)), y=mean_distances))
source_selected = ColumnDataSource(data=dict(x=range(len(unique_dates)), y=mean_distances))
source2 = ColumnDataSource(data=dict(x=range(len(unique_dates)), y=percent_relevant_reviews))
source2_selected = ColumnDataSource(data=dict(x=range(len(unique_dates)), y=percent_relevant_reviews))

tools = 'pan,wheel_zoom,xbox_select,reset'

# Set up plot
plot = figure(plot_height=250, plot_width=1000, title="", tools=tools, active_drag="xbox_select", y_range=(-1,1))
plot.line('x', 'y', source=source, line_width=3, line_alpha=1, line_color='blue', selection_color="red",)
plot.line('x', 'y', source=source_selected, line_width=4, line_alpha=0.8, line_color='red', selection_color="red",)

plot2 = figure(plot_height=250, plot_width=1000, title="", tools=tools, active_drag="xbox_select", y_range=(0,100))
plot2.line('x', 'y', source=source2, line_width=3, line_alpha=1, line_color='green', selection_color="red",)
plot2.line('x', 'y', source=source2_selected, line_width=4, line_alpha=0.8, line_color='red', selection_color="red",)

update_plots_button = Button(label="Update plots", button_type="success")

# Set up widgets
query = TextInput(title="query", value='i love this app')
relevance_threshold = Slider(title="relevance threshold", value=0.5, start=0, end=1, step=0.01)
app_select = Select(title='Select app:', value='eBay (Android)', options=['eBay (Android)', 'YouTube (iOS)'])

reviews = PreText(text='bokeh bokeh', width=1000)
date_range = RangeSlider(title="Select range of dates", start = 1, end = len(unique_dates), step=1, value=(1, len(unique_dates)), width=1000)
update_printed_reviews_button = Button(label="Get relevant reviews in selected timeframe", button_type="success")










def update_app(attrname, old, new):
    selected_app = app_select.value
    if selected_app == 'eBay (Android)':
        pass
    else if selected_app == 'YouTube (iOS)':
        pass
    
    



def update_data():
    

    # Get the current slider values
    q = query.value
    r = relevance_threshold.value
    plot.title.text = 'Average cosine similarity for query: ' + q
    plot2.title.text = '% reviews with cosine similarity above ' + str(r) + ' for query: ' + q
    
    #note - n=10 argument is dumb, get rid of it
    indices, distances = find_relevant_reviews(q, avg_vectors_scaled, final_reviews_unprocessed, scaled=True)
    mean_distances, percent_relevant_reviews = get_topic_evolution_data(distances, q, r)

    source.data = dict(x=range(len(unique_dates)), y=mean_distances)
    source2.data = dict(x=range(len(unique_dates)), y=percent_relevant_reviews)
    
    (start_index, end_index) = date_range.value
    start_index = int(start_index)
    end_index = int(end_index)
    
    #inefficient?percent_relevant_reviews
    source_selected.data = dict(x=range(start_index,end_index), y=mean_distances[start_index:end_index])
    source2_selected.data = dict(x=range(start_index,end_index), y=percent_relevant_reviews[start_index:end_index])
    

update_plots_button.on_click(update_data)
                                                                                                                 
                                                                                                                 
def update_printed_reviews():
    (start_index, end_index) = date_range.value #floats, need to convert to ints
    start_index = int(start_index)
    end_index = int(end_index)
    
    review_indices = [item for sublist in unique_date_indices[start_index:end_index] for item in sublist]
    #print(review_indices)
    
    # Get the current slider values
    q = query.value
    r = relevance_threshold.value


    
    indices, distances = find_relevant_reviews(q, avg_vectors_scaled[review_indices], final_reviews_unprocessed[review_indices], scaled=True)
    
    
    reviews.text = print_relevant_reviews(indices, distances, final_reviews_unprocessed[review_indices], n=10)
    
    original_source_yvals = source.data['y']
    original_source2_yvals = source2.data['y']
    source_selected.data = dict(x=range(start_index,end_index), y=original_source_yvals[start_index:end_index])
    source2_selected.data = dict(x=range(start_index,end_index), y=original_source2_yvals[start_index:end_index])

                                                                               
update_printed_reviews_button.on_click(update_printed_reviews)                                                                                       

# Set up layouts and add to document

col1 = column(query, relevance_threshold,update_plots_button, width=300)
col2 = column(plot, plot2, date_range,update_printed_reviews_button, reviews, width=1000)
layout = row(col1, col2)

curdoc().add_root(layout)
curdoc().title = "App Review GUI!!"



load_time = time.time() - load_start_time
print('Load time:', load_time)