# 2018 Fall-W210-Capstone
# Emergent
*Samir Datta, Thu Nguyen and Yubo Zhang*


Webpage:  https://goldenmonster0602.github.io/w210final.github.io/

**Introduction**


**Data**

For this product demo, we chose to analyze Viber app reviews from the Google Play Store.  First, clean, analyze reviews to find important topics. This includes unsupervised methods for automatic topic clustering, and keyword detection.  And then, map emerging “topics” back to “notable” reviews that contain useful information.


**Lessons Learned**

For this project, we used Latent Dirichlet Allocation (LDA), which is a algorithms used to discover the topics that are present in a corpus. Uniform Manifold Approximation and Projection (UMAP) is a dimension reduction technique that can be used for visualisation. similar with t-SNE.  We also incorporated with Word2Vec, it provides direct access to vector representations of words, which can help achieve decent performance across a variety of tasks. We also tried several methods that unfortunately didn't work out for us, here is a few :


*K-means cluster*: The K-means clustering algorithm is used to find groups which have not been explicitly labeled in the data and to find patterns and make better decisions. However it didn't work with our data.

*PCoA*: Principle Coordinate Analysis is a method to explore and to visualize similarities or dissimilarities of data. It starts with a similarity matrix or dissimilarity matrix and assigns for each item a location in a low-dimensional space. By using PCoA we can visualize individual and/or group differences. However, PCoA didn't work for us. Here is the image of the cluster that we generated.

![alt text](https://github.com/samird121/w210-app-review-capstone/blob/master/new_scraped_reviews/pcoa.png)

[PCoA Page](https://github.com/samird121/w210-app-review-capstone/blob/master/new_scraped_reviews/YuboClusteringtesting.ipynb)





**Reference**
1. Topic Modeling: https://medium.com/nanonets/topic-modeling-with-lsa-psla-lda-and-lda2vec-555ff65b0b05
2. IDEA Method: https://github.com/cuiyungao/IDEA-1
3. Mining Customer Reviews: https://www.cs.uic.edu/~liub/publications/kdd04-revSummary.pdf
4. Modeling Topic Evaluations: http://acsweb.ucsd.edu/~jkalyana/papers/KDD_2015.pdf
