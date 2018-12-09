# 2018 Fall-W210-Capstone
# Emergent
*Samir Datta, Thu Nguyen and Yubo Zhang*


Webpage:  https://goldenmonster0602.github.io/w210final.github.io/

**Introduction**


**Data**

Our data comprises of over 164,000 pre-processed reviews, acquired and used with permission from the ReMine-Lab (2). The reviews come from apps in both the iOS and Google Play store: NOOA Radar, YouTube, Viber, Clean Master, Ebay, and Swiftkey. Included along with the review text is information about the rating, version number, and more. 


**Lessons Learned**

For this project, we used Latent Dirichlet Allocation (LDA), which is a algorithms used to discover the topics that are present in a corpus. Uniform Manifold Approximation and Projection (UMAP) is a dimension reduction technique that can be used for visualisation. similar with t-SNE.  We also incorporated with Word2Vec, it provides direct access to vector representations of words, which can help achieve decent performance across a variety of tasks. We also tried several methods that unfortunately didn't work out for us, here is a few :


*K-means cluster*: The K-means clustering algorithm is used to find groups which have not been explicitly labeled in the data and to find patterns and make better decisions. However it didn't work with our data.

*PCoA*: Principle Coordinate Analysis is a method to explore and to visualize similarities or dissimilarities of data. It starts with a similarity matrix or dissimilarity matrix and assigns for each item a location in a low-dimensional space. By using PCoA we can visualize individual and/or group differences. However, PCoA didn't work for us. Here is the image of the cluster that we generated.

![alt text](https://github.com/samird121/w210-app-review-capstone/blob/master/new_scraped_reviews/pcoa.png)

[PCoA Page](https://github.com/samird121/w210-app-review-capstone/blob/master/new_scraped_reviews/YuboClusteringtesting.ipynb)





**Reference**
1. Topic Modeling: https://medium.com/nanonets/topic-modeling-with-lsa-psla-lda-and-lda2vec-555ff65b0b05
2. Gao, C., Zeng, J., Lyu, M., & King, I. "Online App Review Analysis for Identifying Emerging Issues." *Proceedings of the 40th International Conference on Software Engineering,* 2018
3. Hu, M. & Liu, B. "Mining and Summarizing Customer Reviews." *Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD-2004)*, Aug 22-25, 2004, Seattle, Washington, USA
4. Kalyanam, J., Mantrach, A., Saez-Trumper, D., Vahabi, H. & Lanckriet, G. "Leveraging Social Context for Modeling Topic Evolution." *Proceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining,* Aug 10-13, 2014, Sydney, NSW, Australia
