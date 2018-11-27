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
from sklearn.linear_model import LogisticRegression
#from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import *

from scipy.spatial.distance import cdist, cosine
#from scipy.spatial import cKDTree

from gensim.models import Word2Vec

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from bokeh.io import curdoc, output_file
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource, Range1d, LinearAxis
from bokeh.models.widgets import Slider, TextInput, PreText, Select, RangeSlider, Button, DateRangeSlider
from bokeh.plotting import figure

output_file('app_gui_11-25-18.html')

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

app_names = ['ebay', 'clean_master', 'swiftkey_keyboard', 'viber', 'noaa-radar-pro-severe-weather', 'youtube']
#for testing, it is quicker to just load one app
#app_names = ['ebay']

model_dict = {}
avg_vectors_scaled_dict = {}
scaler_dict = {}
final_reviews_unprocessed_dict = {}
unique_dates_dict = {}
unique_date_indices_dict = {}
final_dates_dict = {}
final_ratings_dict = {}

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
    final_ratings = app_data["ratings"]
    
    unique_dates = np.unique(np.array(final_dates))
    unique_date_indices = []
    for date in unique_dates:
            date_indices = np.where(np.array(final_dates)==date)[0]
            unique_date_indices.append(date_indices)



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
    final_dates_dict[app_name] = final_dates
    final_ratings_dict[app_name] = final_ratings
    
    







def find_relevant_reviews(key, avg_vectors, scaler, model, scaled=False):
    

    distances = None
    
    key_processed = custom_preprocessor(key)[0]
    key_list = key_processed.split(' ')
    #filter to only those in the w2v model's vocabulary
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
    else:
        print("Warning: none of the words in the query were in the model's vocabulary.")
    
    if len(key_list_nonvocab_words) > 0:
        print("Excluded words:", key_list_nonvocab_words)
        
    return distances, key_list_nonvocab_words
    
def print_relevant_reviews(distances, reviews, n=10):
    text = "Most relevant reviews in selected timeframe:\n\n"
    indices = np.argsort(distances) 
    top_indices = indices[-n:]
    for i in range(len(top_indices)):
        index = top_indices[-i-1]
        distance = distances[index]

        text += "'"+str(reviews[index])+", Cosine similarity: "+str(round(distance, 4))+")\n"


    return text
    
def get_app_over_time_data(unique_date_indices, final_ratings):
    
    mean_ratings = []
    n_reviews = []
    
    for date_indices in unique_date_indices:
        date_ratings = np.array(final_ratings)[date_indices]
        mean_ratings.append(np.mean(date_ratings))
        
        n_reviews.append(len(date_indices))
        
       
    return mean_ratings, n_reviews
    
def get_topic_evolution_data(distances, unique_date_indices, relevance_threshold = 0.8):
    
    
    mean_distances = []
    n_relevant_reviews = []
    percent_relevant_reviews= []
    
    for date_indices in unique_date_indices:
        date_distances = np.array(distances)[date_indices]
        mean_distances.append(np.mean(date_distances))
        
        relevant_indeces = np.where(date_distances > relevance_threshold)[0]
        n = len(relevant_indeces)
        percent = 100*(n/len(date_indices))

        n_relevant_reviews.append(n)
        percent_relevant_reviews.append(percent)
       
    return mean_distances, percent_relevant_reviews


#this is too slow without pre-vectorizing words which may just not be worth it
def get_timeframe_words(final_reviews_unprocessed, review_indices):
    print('nb start')
    cv = CountVectorizer(min_df=1, ngram_range=(0,2))
    cv_sentences = cv.fit_transform(final_reviews_unprocessed)
    n_reviews = len(final_reviews_unprocessed)
    y = [n in review_indices for n in range(0, n_reviews)]
    
    lr = LogisticRegression()
    lr.fit(cv_sentences, y)
    topweight_indeces = np.argsort(lr.coef_[0])[-20:]
    topweights = lr.coef_[0][topweight_indeces]
    cv_featurenames = cv.get_feature_names()
    
    #preds = nb.predict(cv_sentences[review_indices])
    #top_pred_indices = np.argsort(preds)
    #top_pred_indices = top_pred_indices[-10:]
    
    output = ''
    for index in topweight_indeces:
        output = output + cv_featurenames[index] + ', ' + str(lr.coef_[0][index])
        output += '\n'
    print('nb end')
    return output

def get_timeframe_words2(final_reviews_unprocessed, avg_vectors_scaled, review_indices):
    print('nb start')

    n_reviews = len(final_reviews_unprocessed)
    y = [n in review_indices for n in range(0, n_reviews)]
    
    lr = LogisticRegression()
    lr.fit(avg_vectors_scaled, y)
    preds = lr.predict(avg_vectors_scaled[review_indices])
    top_pred_indices = np.argsort(preds)
    top_pred_indices = top_pred_indices[-10:]
    
    output = ''
    for index in top_pred_indices:
        output += final_reviews_unprocessed[review_indices][index]
        output += '\n'
    print('nb end')
    return output
        

    

    
    
    
    
#initialize variables for gui
q = 'i love this app'
initial_app_name = 'ebay'

model = model_dict[initial_app_name]
avg_vectors_scaled = avg_vectors_scaled_dict[initial_app_name]
scaler = scaler_dict[initial_app_name]
final_reviews_unprocessed = final_reviews_unprocessed_dict[initial_app_name]
unique_dates = unique_dates_dict[initial_app_name]
unique_date_indices = unique_date_indices_dict[initial_app_name]
final_ratings = final_ratings_dict[initial_app_name]
r = 0.5

distances, key_list_nonvocab_words = find_relevant_reviews(q, avg_vectors_scaled, scaler, model, scaled=True)
mean_distances, percent_relevant_reviews = get_topic_evolution_data(distances, unique_date_indices, r)
printed_reviews_text = print_relevant_reviews(distances, final_reviews_unprocessed, n=10)
mean_ratings, n_reviews = get_app_over_time_data(unique_date_indices, final_ratings)
    
    
    
#BOKEH!

col1_width=500
col2_width=800
    
    
    
    
# Set up data

source = ColumnDataSource(data=dict(x=unique_dates, y=mean_ratings, y2=n_reviews))
#source_selected = ColumnDataSource(data=dict(x=unique_dates, y=mean_ratings))
source2 = ColumnDataSource(data=dict(x=unique_dates, y=percent_relevant_reviews))
source2_selected = ColumnDataSource(data=dict(x=unique_dates, y=percent_relevant_reviews))

tools = 'pan,wheel_zoom,xbox_select,reset'



# Set up plots and widgets
#COLUMN 1
app_select = Select(title='Select app:', value=app_names[0], options=app_names)
query = TextInput(title="query", value='i love this app')
words_not_in_vocab = PreText(text='Query words not in vocab: '+', '.join(key_list_nonvocab_words))
relevance_threshold = Slider(title="relevance threshold", value=0.5, start=0, end=1, step=0.01)
limit_to_keyword = TextInput(title="limit to reviews containing the word:", value='')
limit_to_ratings = RangeSlider(title="limit to reviews with ratings:", start = 1, end = 5, step=1, value=(1,5), width=col1_width)
update_plots_button = Button(label="Update everything!", button_type="success")

#COLUMN 2
plot = figure(plot_height=250, plot_width=col2_width, title='Average ratings for app: ' + initial_app_name, tools=tools, active_drag="xbox_select", y_range=(1,5), x_axis_type='datetime')
plot.extra_y_ranges = {"n_reviews": Range1d(start=0, end=np.max(n_reviews))}
plot.vbar('x', top='y2', source=source, color="purple",width=0.9, y_range_name="n_reviews", legend="# reviews per day")
plot.add_layout(LinearAxis(y_range_name="n_reviews"), 'right')
plot.line('x', 'y', source=source,  line_width=2, line_alpha=0.8, line_color='blue', legend="Mean ratings per day")


plot2 = figure(plot_height=250, plot_width=col2_width, title='% reviews with cosine similarity above ' + str(r) + ' for query: ' + q, y_range=(0,100), x_axis_type='datetime')
plot2.line('x', 'y', source=source2, line_width=3, line_alpha=1, line_color='green', selection_color="red",)
plot2.line('x', 'y', source=source2_selected, line_width=4, line_alpha=0.8, line_color='red', selection_color="red",)

date_range = DateRangeSlider(title="Select range of dates", start = unique_dates[0], end = unique_dates[-1], step=1, value=(unique_dates[0], unique_dates[-1]), width=col2_width)
reviews = PreText(text=printed_reviews_text, width=col2_width)

#update_printed_reviews_button = Button(label="Get relevant reviews in selected timeframe", button_type="success")
#update_timeframe_words = Button(label="Find phrases used more often in selected timeframe", button_type="success")
#timeframe_words = PreText(text='Select a range of dates to list the words most specifically associated with those dates.', width=col2_width)








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
    final_ratings = final_ratings_dict[initial_app_name]
    final_dates = final_dates_dict[initial_app_name]
    
    
    (start_rating, end_rating) = limit_to_ratings.value
    #if a range of ratings has been selected
    if start_rating != 1 or end_rating != 5:
        ratings_indeces = []
        for i in range(start_rating, end_rating+1):
            i_indeces = np.where(final_ratings == i)[0]
            for i_index in i_indeces:
                ratings_indeces.append(i_index)
            
        #flatten array
        #ratings_indeces = [item for sublist in ratings_indeces for item in sublist]
        print(ratings_indeces)
        avg_vectors_scaled = avg_vectors_scaled[ratings_indeces]
        final_reviews_unprocessed = final_reviews_unprocessed[ratings_indeces]
        final_ratings = final_ratings[ratings_indeces]
        
        final_dates = final_dates[ratings_indeces]
        unique_dates = np.unique(np.array(final_dates)) 
        unique_date_indices = []
        for date in unique_dates:
                date_indices = np.where(np.array(final_dates)==date)[0]
                unique_date_indices.append(date_indices)
                
        unique_dates = [parser.parse(date) for date in unique_dates]
        
        
    

    # Get the current slider values
    q = query.value
    r = relevance_threshold.value
    plot.title.text = 'Average ratings for app: ' + app_name
    plot2.title.text = '% reviews with cosine similarity above ' + str(r) + ' for query: ' + q
    


    distances, key_list_nonvocab_words = find_relevant_reviews(q, avg_vectors_scaled, scaler, model, scaled=True)
    words_not_in_vocab.text = 'Query words not in vocab: '+', '.join(key_list_nonvocab_words)

    mean_distances, percent_relevant_reviews = get_topic_evolution_data(distances, unique_date_indices, r)
    mean_ratings, n_reviews = get_app_over_time_data(unique_date_indices, final_ratings)

    source.data = dict(x=unique_dates, y=mean_ratings, y2=n_reviews)
    source2.data = dict(x=unique_dates, y=percent_relevant_reviews)
    
    
    
    
    (start_date, end_date) = date_range.value_as_datetime
    min_date = date_range.start
    max_date = date_range.end
    
    start_index = np.where(np.array(unique_dates, dtype='datetime64') >= np.datetime64(start_date))[0][0]
    end_index = np.where(np.array(unique_dates, dtype='datetime64') <= np.datetime64(end_date))[0][-1]
    
    #if a range of dates has been selected (i.e. the selection isn't just on the min/max)
    if start_date != min_date or end_date != max_date:        
        review_indices = [item for sublist in unique_date_indices[start_index:end_index] for item in sublist]
        print_distances = np.array(distances)[review_indices]
        print_reviews_unprocessed = final_reviews_unprocessed[review_indices]
        
        #timeframe_words.text = get_timeframe_words(final_reviews_unprocessed, review_indices)
    else:
        print_distances = distances
        print_reviews_unprocessed = final_reviews_unprocessed
        #timeframe_words.text='Select a range of dates to list the words most specifically associated with those dates.'
    
    reviews.text = print_relevant_reviews(print_distances, print_reviews_unprocessed, n=10)
    
    #original_source_xvals = source.data['x']
    original_source2_xvals = source2.data['x']
    #original_source_yvals = source.data['y']
    original_source2_yvals = source2.data['y']
    
    #source_selected.data = dict(x=original_source_xvals[start_index:end_index], y=original_source_yvals[start_index:end_index])
    source2_selected.data = dict(x=original_source2_xvals[start_index:end_index], y=original_source2_yvals[start_index:end_index])
    
    


update_plots_button.on_click(update_data)                                                                           
#update_printed_reviews_button.on_click(update_data) 




# Set up layouts and add to document

col1 = column(app_select, query,words_not_in_vocab, relevance_threshold,limit_to_keyword,limit_to_ratings, update_plots_button, width=col1_width)
col2 = column(plot, plot2, date_range,reviews, width=col2_width)
layout = row(col1, col2)

curdoc().add_root(layout)
curdoc().title = "App Review GUI!!"




load_time = time.time() - load_start_time
print('Load time:', load_time)