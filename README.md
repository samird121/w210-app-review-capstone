# 2018 fall-W210-app-review-capstone Emergent
Samir Datta, Thu Nguyen and Yubo Zhang


Webpage:  https://goldenmonster0602.github.io/w210final.github.io/

Introduction 


Data

For this product demo, we chose to analyze Viber app reviews from the Google Play Store.  First, clean, analyze reviews to find important topics. This includes unsupervised methods for automatic topic clustering, and keyword detection.  And then, map emerging “topics” back to “notable” reviews that contain useful information.


Lessons Learned

For this project, we used Latent Dirichlet Allocation (LDA), which is a algorithms used to discover the topics that are present in a corpus. We also incorporated with Word2Vec, it provides direct access to vector representations of words, which can help achieve decent performance across a variety of tasks. We also tried several methods that unfortunately didn't work out for us, here is a few :

K-means cluster: The K-means clustering algorithm is used to find groups which have not been explicitly labeled in the data and to find patterns and make better decisions. However it didn't work with our data.

PCoA 
Principle Coordinate Analysis is a method to explore and to visualize similarities or dissimilarities of data. It starts with a similarity matrix or dissimilarity matrix and assigns for each item a location in a low-dimensional space. By using PCoA we can visualize individual and/or group differences. 

![alternativetext](w210-app-review-capstone/new_scraped_reviews/pcoa.png)





Reference
