print('Loading GUI...')

import time
load_start_time = time.time()

import csv
import nltk
import re
import numpy as np
import pandas as pd
from dateutil import parser

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
from bokeh.models.widgets import Slider, TextInput, PreText, Select, RangeSlider, Button, DateRangeSlider
from bokeh.plotting import figure

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








######

app_names = ['ebay', 'clean_master', 'swiftkey_keyboard', 'viber']
#for testing, it is quicker to just load one app
#app_names = ['ebay']

model_dict = {}
avg_vectors_scaled_dict = {}
scaler_dict = {}
final_reviews_unprocessed_dict = {}
unique_dates_dict = {}
unique_date_indices_dict = {}


#load up the data
for app_name in app_names:
    print('Loading data for ', app_name, '.......')
    
    data_filename = "../../large files/"+app_name+"_post_processing_data.csv"
    model_filename = "../../large files/"+app_name+".model"
    
    app_data = pd.read_csv(data_filename)
    final_reviews_unprocessed = app_data["reviews_unprocessed"]
    final_dates = app_data["dates"]
    final_versions = app_data["versions"]
    final_reviews_processed = app_data["reviews_processed"]

    final_reviews =  [review.split(" ") for review in final_reviews_processed]
    #final_reviews_unprocessed =  np.array(reviews)[nonzero_indeces]
    #final_ratings = np.array(ratings)[nonzero_indeces]
    #final_titles = np.array(titles)[nonzero_indeces]
    #final_dates = np.array(dates)[nonzero_indeces]
    unique_dates = np.unique(np.array(final_dates))
    #print(unique_dates)
    unique_date_indices = []
    for date in unique_dates:
            date_indices = np.where(np.array(final_dates)==date)[0]
            unique_date_indices.append(date_indices)
    #final_versions = np.array(versions)[nonzero_indeces]


    model = Word2Vec.load(model_filename)

    #here we create a vector for each review,
    #which will be the simple average of all word vectors in that review.
    #these vectors will then be used for clustering, data reduction, etc.
    avg_vectors = []
    for review in final_reviews:
        avg_vectors.append(np.mean([model.wv[word] for word in review], axis=0))

    avg_vectors = np.array(avg_vectors)  

    #scaling 
    scaler = StandardScaler()
    avg_vectors_scaled = scaler.fit_transform(avg_vectors)
    
    model_dict[app_name] = model
    avg_vectors_scaled_dict[app_name] = avg_vectors_scaled
    scaler_dict[app_name] = scaler
    final_reviews_unprocessed_dict[app_name] = np.array(final_reviews_unprocessed.tolist())
    unique_dates_dict[app_name] = [parser.parse(date) for date in unique_dates]
    unique_date_indices_dict[app_name] = unique_date_indices
    
    







def find_relevant_reviews(key, avg_vectors, raw_text, scaler, model, scaled=False):
    
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
        
    return indices, distances, key_list_nonvocab_words
    
def print_relevant_reviews(indices, distances, reviews, n=10):
    text = "Most relevant reviews in selected timeframe:\n\n"
    top_indices = indices[-n:]
    for i in range(len(top_indices)):
        index = top_indices[-i-1]
        distance = distances[index]

        text += "'"+str(reviews[index])+", Cosine similarity: "+str(round(distance, 4))+")\n"

        #text += "'"+str(final_reviews_unprocessed[index])+"' (Rating: "+str(final_ratings[index])+")\n"

    return text
    

    
def get_topic_evolution_data(distances, query, unique_date_indices, relevance_threshold = 0.8):
    
    
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
        
        #date_ratings = np.array(final_ratings)[date_indices][relevant_indeces]
        #mean_ratings.append(np.mean(date_ratings))
       
    return mean_distances, percent_relevant_reviews

    

    
    
    
    
#initialize variables for gui
query = 'i love this app'

initial_app_name = 'ebay'

model = model_dict[initial_app_name]
avg_vectors_scaled = avg_vectors_scaled_dict[initial_app_name]
scaler = scaler_dict[initial_app_name]
final_reviews_unprocessed = final_reviews_unprocessed_dict[initial_app_name]
unique_dates = unique_dates_dict[initial_app_name]
unique_date_indices = unique_date_indices_dict[initial_app_name]


indices, distances, key_list_nonvocab_words = find_relevant_reviews(query, avg_vectors_scaled, final_reviews_unprocessed,  scaler, model, scaled=True)
mean_distances, percent_relevant_reviews = get_topic_evolution_data(distances, query, unique_date_indices, relevance_threshold = 0.5)


#indices, distances = find_relevant_reviews(query, avg_vectors, final_reviews_unprocessed, n=10, scaled=False)
#for i in range(len(indices)):
#    index = indices[-(i+1)]
#    print(final_reviews_unprocessed[index])
#    print("Rating:", final_ratings[index])
#    print("Cosine similarity:", round(distances[index], 4))
#    print('*******\n')
    
    
    
    
    
    
    
    
    
#BOKEH!

    
    
    
    
    
# Set up data

source = ColumnDataSource(data=dict(x=unique_dates, y=mean_distances))
source_selected = ColumnDataSource(data=dict(x=unique_dates, y=mean_distances))
source2 = ColumnDataSource(data=dict(x=unique_dates, y=percent_relevant_reviews))
source2_selected = ColumnDataSource(data=dict(x=unique_dates, y=percent_relevant_reviews))

tools = 'pan,wheel_zoom,xbox_select,reset'

# Set up plot
plot = figure(plot_height=250, plot_width=1000, title="", tools=tools, active_drag="xbox_select", y_range=(-1,1), x_axis_type='datetime')
plot.line('x', 'y', source=source,  line_width=3, line_alpha=1, line_color='blue', selection_color="red",)
plot.line('x', 'y', source=source_selected, line_width=4, line_alpha=0.8, line_color='red', selection_color="red",)

plot2 = figure(plot_height=250, plot_width=1000, title="", tools=tools, active_drag="xbox_select", y_range=(0,100), x_axis_type='datetime')
plot2.line('x', 'y', source=source2, line_width=3, line_alpha=1, line_color='green', selection_color="red",)
plot2.line('x', 'y', source=source2_selected, line_width=4, line_alpha=0.8, line_color='red', selection_color="red",)

update_plots_button = Button(label="Update plots", button_type="success")

# Set up widgets
query = TextInput(title="query", value='i love this app')
words_not_in_vocab = PreText(text='Query words not in vocab: '+', '.join(key_list_nonvocab_words))
relevance_threshold = Slider(title="relevance threshold", value=0.5, start=0, end=1, step=0.01)
app_select = Select(title='Select app:', value=app_names[0], options=app_names)

reviews = PreText(text='', width=1000)
#date_range = RangeSlider(title="Select range of dates", start = 1, end = len(unique_dates), step=1, value=(1, len(unique_dates)), width=1000)
date_range = DateRangeSlider(title="Select range of dates", start = unique_dates[0], end = unique_dates[-1], step=1, value=(unique_dates[0], unique_dates[-1]), width=1000)
update_printed_reviews_button = Button(label="Get relevant reviews in selected timeframe", button_type="success")










def update_app(attrname, old, new):
    
    app_name = app_select.value
    unique_dates = unique_dates_dict[app_name]
    
    #just to reset the slider
    date_range.start = unique_dates[0]
    date_range.end = unique_dates[-1]
    date_range.value = (unique_dates[0], unique_dates[-1])
    
    #do the rest here
    update_data()

   

app_select.on_change('value', update_app)



def update_data():
    #get values specific to the currently selected app
    app_name = app_select.value
    
    model = model_dict[app_name]
    avg_vectors_scaled = avg_vectors_scaled_dict[app_name]
    scaler = scaler_dict[app_name]
    final_reviews_unprocessed = final_reviews_unprocessed_dict[app_name]
    unique_dates = unique_dates_dict[app_name]
    unique_date_indices = unique_date_indices_dict[app_name]
    
    #date_range.start = 1
    #date_range.end = len(unique_dates)
    

    # Get the current slider values
    q = query.value
    r = relevance_threshold.value
    plot.title.text = 'Average cosine similarity for query: ' + q
    plot2.title.text = '% reviews with cosine similarity above ' + str(r) + ' for query: ' + q
    
    #note - n=10 argument is dumb, get rid of it
    indices, distances, key_list_nonvocab_words = find_relevant_reviews(q, avg_vectors_scaled, final_reviews_unprocessed, scaler, model, scaled=True)
    words_not_in_vocab.text = 'Query words not in vocab: '+', '.join(key_list_nonvocab_words)
    
    mean_distances, percent_relevant_reviews = get_topic_evolution_data(distances, q,  unique_date_indices, r)
    

    source.data = dict(x=unique_dates, y=mean_distances)
    source2.data = dict(x=unique_dates, y=percent_relevant_reviews)
    
    #(start_index, end_index) = date_range.value
    #start_index = int(start_index)
    #end_index = int(end_index)
    
    #inefficient?percent_relevant_reviews
    #source_selected.data = dict(x=unique_dates, y=mean_distances[start_index:end_index])
    #source2_selected.data = dict(x=unique_dates, y=percent_relevant_reviews[start_index:end_index])
    update_printed_reviews()


update_plots_button.on_click(update_data)

                                                                                                                 
                                                                                                                 
def update_printed_reviews():
    
    app_name = app_select.value
    
    #get app data
    model = model_dict[app_name]
    avg_vectors_scaled = avg_vectors_scaled_dict[app_name]
    scaler = scaler_dict[app_name]
    final_reviews_unprocessed = final_reviews_unprocessed_dict[app_name]
    unique_dates = unique_dates_dict[app_name]
    unique_date_indices = unique_date_indices_dict[app_name]

    (start_date, end_date) = date_range.value_as_datetime
    
    start_index = np.where(np.array(unique_dates, dtype='datetime64') >= start_date)[0][0]
    end_index = np.where(np.array(unique_dates, dtype='datetime64') <= end_date)[0][-1]

    review_indices = [item for sublist in unique_date_indices[start_index:end_index] for item in sublist]
    
    
   

    
    # Get the current slider values
    q = query.value
    r = relevance_threshold.value


    
    indices, distances, key_list_nonvocab_words = find_relevant_reviews(q, avg_vectors_scaled[review_indices], final_reviews_unprocessed[review_indices], scaler, model,scaled=True)


    words_not_in_vocab.text = 'Query words not in vocab: '+', '.join(key_list_nonvocab_words)
    reviews.text = print_relevant_reviews(indices, distances, final_reviews_unprocessed[review_indices], n=10)
    
    original_source_xvals = source.data['x']
    original_source2_xvals = source2.data['x']
    original_source_yvals = source.data['y']
    original_source2_yvals = source2.data['y']
    
    source_selected.data = dict(x=original_source_xvals[start_index:end_index], y=original_source_yvals[start_index:end_index])
    source2_selected.data = dict(x=original_source2_xvals[start_index:end_index], y=original_source2_yvals[start_index:end_index])

                                                                               
update_printed_reviews_button.on_click(update_printed_reviews)                                                                                       

# Set up layouts and add to document

col1 = column(app_select, query,words_not_in_vocab, relevance_threshold,update_plots_button, width=300)
col2 = column(plot, plot2, date_range,update_printed_reviews_button, reviews, width=1000)
layout = row(col1, col2)

curdoc().add_root(layout)
curdoc().title = "App Review GUI!!"



load_time = time.time() - load_start_time
print('Load time:', load_time)