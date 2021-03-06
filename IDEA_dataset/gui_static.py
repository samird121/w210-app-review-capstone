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

from bokeh.io import curdoc, output_file, show
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource, Range1d, LinearAxis, CustomJS, HoverTool, Title
from bokeh.models.widgets import Slider, TextInput, PreText, Select, RangeSlider, Button, DateRangeSlider
from bokeh.plotting import figure

output_file('bokeh_gui_for_website_12-8-18.html')

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

#app_names = ['ebay', 'clean_master', 'swiftkey_keyboard', 'viber', 'noaa-radar-pro-severe-weather', 'youtube']
#for testing, it is quicker to just load one app
app_names = ['viber', 'ebay', 'youtube']


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
    
def print_relevant_reviews(distances, ratings, reviews, n=10):
    text = "Most relevant reviews in selected timeframe:\n\n"
    indices = np.argsort(distances) 
    top_indices = indices[-n:]
    for i in range(len(top_indices)):
        index = top_indices[-i-1]
        distance = distances[index]
        rating = ratings[index]
        text += str(reviews[index])
        text += " (Rating: "+str(int(rating))+"/5 stars)\n"


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

query_options = ['update', 'download','freeze', 'crash', 'load', 'buffer', 'video', 'version',  'quality', 'connection', 'good', 'bad', 'love', 'hate']
#query_options = ['good', 'bad']
initial_app_name = 'viber'
#app_name_options = ['ebay']


r = 0.25

percent_relevant_reviews_dict = {}
printed_reviews_text_dict = {}
mean_ratings_dict = {}
n_reviews_dict = {}
datestrings_dict = {}
for app_name in app_names:
    percent_relevant_reviews_dict[app_name] = {}
    printed_reviews_text_dict[app_name] = {}

for app_name in app_names:
    print('Generating data for '+app_name+'...')
    model = model_dict[app_name]
    avg_vectors_scaled = avg_vectors_scaled_dict[app_name]
    scaler = scaler_dict[app_name]
    final_reviews_unprocessed = final_reviews_unprocessed_dict[app_name]
    unique_dates = unique_dates_dict[app_name]
    unique_date_indices = unique_date_indices_dict[app_name]
    final_ratings = final_ratings_dict[app_name]
    
    for q in query_options:
        print('query: '+q)
        distances, key_list_nonvocab_words = find_relevant_reviews(q, avg_vectors_scaled, scaler, model, scaled=True)
        mean_distances, percent_relevant_reviews = get_topic_evolution_data(distances, unique_date_indices, r)
        printed_reviews_text = print_relevant_reviews(distances,final_ratings, final_reviews_unprocessed, n=10)
        mean_ratings, n_reviews = get_app_over_time_data(unique_date_indices, final_ratings)
        percent_relevant_reviews_dict[app_name][q] = percent_relevant_reviews
        printed_reviews_text_dict[app_name][q] = [printed_reviews_text]
        
        
        
    mean_ratings_dict[app_name] = mean_ratings
    n_reviews_dict[app_name] = n_reviews
    datestrings = [str(d)[:10]for d in np.array(unique_dates, dtype=np.datetime64)]
    datestrings_dict[app_name] = datestrings
    
    
    
    
#BOKEH!
col1_width=500
col2_width=800
#print(percent_relevant_reviews_dict) 
    
    
    
# Set up data

percent_relevant_reviews = percent_relevant_reviews_dict[initial_app_name][query_options[0]]
unique_dates = unique_dates_dict[initial_app_name]
mean_ratings = mean_ratings_dict[initial_app_name]
n_reviews = n_reviews_dict[initial_app_name]
printed_reviews_text = printed_reviews_text_dict[initial_app_name][query_options[0]][0]

datestrings = datestrings_dict[initial_app_name]

source = ColumnDataSource(data=dict(x=unique_dates, y=mean_ratings, y2=n_reviews, dates=datestrings))
source2 = ColumnDataSource(data=dict(x=unique_dates, y=percent_relevant_reviews, dates=datestrings))
source2_selected = ColumnDataSource(data=dict(x=unique_dates, y=percent_relevant_reviews, dates=datestrings))





#[app_name][query]
prr_source_ebay = ColumnDataSource(data=percent_relevant_reviews_dict['ebay'])
prr_source_viber = ColumnDataSource(data=percent_relevant_reviews_dict['viber'])
prr_source_youtube = ColumnDataSource(data=percent_relevant_reviews_dict['youtube'])

prt_source_ebay = ColumnDataSource(data=printed_reviews_text_dict['ebay'])
prt_source_viber = ColumnDataSource(data=printed_reviews_text_dict['viber'])
prt_source_youtube = ColumnDataSource(data=printed_reviews_text_dict['youtube'])

#[app_name]
ud_source_ebay = ColumnDataSource(data=dict(x=unique_dates_dict['ebay']))
ud_source_viber = ColumnDataSource(data=dict(x=unique_dates_dict['viber']))
ud_source_youtube = ColumnDataSource(data=dict(x=unique_dates_dict['youtube']))

mr_source_ebay = ColumnDataSource(data=dict(x=mean_ratings_dict['ebay']))
mr_source_viber = ColumnDataSource(data=dict(x=mean_ratings_dict['viber']))
mr_source_youtube = ColumnDataSource(data=dict(x=mean_ratings_dict['youtube']))

nr_source_ebay = ColumnDataSource(data=dict(x=n_reviews_dict['ebay']))
nr_source_viber = ColumnDataSource(data=dict(x=n_reviews_dict['viber']))
nr_source_youtube = ColumnDataSource(data=dict(x=n_reviews_dict['youtube']))

dates_source_ebay = ColumnDataSource(data=dict(x=datestrings_dict['ebay']))
dates_source_viber = ColumnDataSource(data=dict(x=datestrings_dict['viber']))
dates_source_youtube = ColumnDataSource(data=dict(x=datestrings_dict['youtube']))


#tools = 'pan,wheel_zoom,xbox_select,reset'







# Set up plots and widgets
#COLUMN 1
app_select = Select(title='Select app:', value=app_names[0], options=app_names)
#query = TextInput(title="query", value='i love this app')
query = Select(title='Select query:', value=query_options[0], options=query_options)



limit_to_keyword = TextInput(title="limit to reviews containing the word:", value='')
update_plots_button = Button(label="Update everything!", button_type="success")

#COLUMN 2
plot = figure(plot_height=250, plot_width=col2_width, title='Average Ratings and Number of Reviews for App', y_range=(1,5), x_axis_type='datetime', toolbar_location=None)
plot.add_tools(HoverTool(tooltips=[( 'Date', '@dates'),( 'Mean rating',  '@y' ), ( '# reviews', '@y2'      )],mode='vline'))
plot.title.align = 'center'
plot.add_layout(Title(text=" Mean Rating ", align="center", background_fill_color ="blue", text_color="white"), "left")
plot.yaxis.ticker = [1, 2, 3, 4, 5]
plot.yaxis.major_label_overrides = {1: '  1', 2: '  2', 3: '  3', 4: '  4', 5: '  5'}

plot.extra_y_ranges = {"n_reviews": Range1d(start=0, end=np.max(n_reviews))}
plot.vbar('x', top='y2', source=source, color="purple",width=0.9, y_range_name="n_reviews")
plot.add_layout(LinearAxis(y_range_name="n_reviews"), 'right')
plot.line('x', 'y', source=source,  line_width=2, line_alpha=0.8, line_color='blue')
plot.add_layout(Title(text=" Number of Reviews ", align="center", background_fill_color ="purple", text_color="white"), "right")

plot2 = figure(plot_height=250, plot_width=col2_width, title='% Reviews Relevant to Query', y_range=(0,100), x_axis_type='datetime', toolbar_location=None)
plot2.add_tools(HoverTool(tooltips=[( 'Date', '@dates'),( '% relevant reviews',  '@y')],mode='vline'))

plot2.line('x', 'y', source=source2, line_width=3, line_alpha=1, line_color='red')
plot2.title.align = 'center'
plot2.add_layout(Title(text=" % relevant reviews ", align="center", background_fill_color ="red", text_color="white"), "left")

reviews = PreText(text=printed_reviews_text, width=col1_width)

#update_printed_reviews_button = Button(label="Get relevant reviews in selected timeframe", button_type="success")
#update_timeframe_words = Button(label="Find phrases used more often in selected timeframe", button_type="success")
#timeframe_words = PreText(text='Select a range of dates to list the words most specifically associated with those dates.', width=col2_width)

#javascript callbacks for standalone html files
callback = CustomJS(args=dict(source=source, source2=source2,reviews=reviews,plot=plot, plot2=plot2, prr_source_ebay=prr_source_ebay, prr_source_viber=prr_source_viber, prr_source_youtube=prr_source_youtube,prt_source_ebay=prt_source_ebay, prt_source_viber=prt_source_viber, prt_source_youtube=prt_source_youtube, ud_source_ebay=ud_source_ebay, ud_source_viber=ud_source_viber,ud_source_youtube=ud_source_youtube, app_select=app_select, query=query, mr_source_ebay=mr_source_ebay, mr_source_viber=mr_source_viber, mr_source_youtube=mr_source_youtube, nr_source_ebay=nr_source_ebay, nr_source_viber=nr_source_viber, nr_source_youtube=nr_source_youtube, dates_source_ebay=dates_source_ebay, dates_source_viber=dates_source_viber, dates_source_youtube=dates_source_youtube), code="""
    var data2 = source2.data;
    var q = query.value
    
    var current_app = app_select.value
    
    if (current_app == 'ebay'){
        var newydata = prr_source_ebay.data;
        var newxdata = ud_source_ebay.data;
        
        var newmrdata = mr_source_ebay.data;
        var newnrdata = nr_source_ebay.data;
        
        var newtextdata = prt_source_ebay.data;
        
        var newdatedata = dates_source_ebay.data;
        
    } else if (current_app == 'viber'){
        var newydata = prr_source_viber.data;
        var newxdata = ud_source_viber.data;
        
        var newmrdata = mr_source_viber.data;
        var newnrdata = nr_source_viber.data;
        
        var newtextdata = prt_source_viber.data;
        
        var newdatedata = dates_source_viber.data;
        
    } else if (current_app == 'youtube'){
        var newydata = prr_source_youtube.data;
        var newxdata = ud_source_youtube.data;
        
        var newmrdata = mr_source_youtube.data;
        var newnrdata = nr_source_youtube.data;
        
        var newtextdata = prt_source_youtube.data;
        
        var newdatedata = dates_source_youtube.data;
        
    }

    
    data2['x'] = newxdata['x']
    data2['y'] = newydata[q]
    data2['dates'] = newdatedata['x']
        
    source2.change.emit();
    
    
    var data = source.data;
    
    data['x'] = newxdata['x']
    data['y'] = newmrdata['x']
    data['y2'] = newnrdata['x']
    data['dates'] = newdatedata['x']
    
    source.change.emit();
    
    
    reviews.text = newtextdata[q][0];
    
""")

#for (var i = 0; i < x.length; i++) {
#        y[i] = newy[i]
#    }


#query.js_on_change('value', callback)
update_plots_button.js_on_click(callback)





def update_app(attrname, old, new):
    
    app_name = app_select.value
    unique_dates = unique_dates_dict[app_name]
    
    #just to reset the slider
    date_range.start = unique_dates[0]
    date_range.end = unique_dates[-1]
    date_range.value = (unique_dates[0], unique_dates[-1])
    
    #do the rest here
    update_data()
    
    
    
#app_select.on_change('value', update_app)



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
    
    

        
        
    

    # Get the current slider values
    q = query.value
    r = relevance_threshold.value
    plot.title.text = 'Average ratings for app: ' + app_name
    plot2.title.text = '% reviews relevant to query: ' + q
    


    #distances, key_list_nonvocab_words = find_relevant_reviews(q, avg_vectors_scaled, scaler, model, scaled=True)
    #words_not_in_vocab.text = 'Query words not in vocab: '+', '.join(key_list_nonvocab_words)

    #mean_distances, percent_relevant_reviews = get_topic_evolution_data(distances, unique_date_indices, r)
    #mean_ratings, n_reviews = get_app_over_time_data(unique_date_indices, final_ratings)

    #source.data = dict(x=unique_dates, y=mean_ratings, y2=n_reviews)
    percent_relevant_reviews = percent_relevant_reviews_dict[app_name][q]
    source2.data = dict(x=unique_dates, y=percent_relevant_reviews)
    
    
    
    
    
    
    print_distances = distances
    print_reviews_unprocessed = final_reviews_unprocessed
    reviews.text = print_relevant_reviews(print_distances, print_reviews_unprocessed, n=10)
    

    


#update_plots_button.on_click(update_data)                                                                           
#update_printed_reviews_button.on_click(update_data) 




# Set up layouts and add to document

col1 = column(app_select, query, update_plots_button, reviews, width=col1_width)
col2 = column(plot, plot2, width=col2_width)
layout = row(col1, col2)

#curdoc().add_root(layout)
#curdoc().title = "App Review GUI!!"


show(layout)

load_time = time.time() - load_start_time
print('Load time:', load_time)